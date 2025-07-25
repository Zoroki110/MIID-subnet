# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): YANEZ - MIID Team
# Copyright © 2025 YANEZ

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Name Variation Miner Module

This module implements a Bittensor miner that generates alternative spellings for names
using a remote LLM service (Chutes).
######### Requires a CHUTES_API_KEY environment variable ########
The miner receives requests from validators containing
a list of names and a query template, processes each name through the LLM, extracts
the variations from the LLM's response, and returns them to the validator.

The miner follows these steps:
1. Receive a request with names and a query template
2. For each name, query the LLM to generate variations
3. Process the LLM responses to extract clean variations
4. Return the variations to the validator

The processing logic handles different response formats from LLMs, including:
- Comma-separated lists
- Line-separated lists
- Space-separated lists with numbering

For debugging and analysis, the miner also saves:
- Raw LLM responses
- Processed variations in JSON format
- A pandas DataFrame with the variations

Each mining run is saved with a unique timestamp identifier to distinguish between
different runs and facilitate analysis of results over time.
"""

import time
import typing
import asyncio
import bittensor as bt
import aiohttp
import pandas as pd
import os
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

# Bittensor Miner Template:
from MIID.protocol import IdentitySynapse

# import base miner class which takes care of most of the boilerplate
from MIID.base.miner import BaseMinerNeuron


class Miner(BaseMinerNeuron):
    """
    Name Variation Miner Neuron
    
    This miner receives requests from validators to generate alternative spellings for names,
    and responds with variations generated using the Chutes API.
    
    The miner handles the following tasks:
    - Processing incoming requests for name variations
    - Querying a remote LLM to generate variations
    - Extracting and cleaning variations from LLM responses
    - Returning the processed variations to the validator
    - Saving intermediate results for debugging and analysis
    
    Each mining run is saved with a unique timestamp identifier to distinguish between
    different runs and facilitate analysis of results over time.
    
    Configuration:
    - model_name: The Chutes model to use (default: 'tinyllama:latest')
    - output_path: Directory for saving mining results (default: logging_dir/mining_results)
    """

    def __init__(self, config=None):
        """
        Initialize the Name Variation Miner.
        
        Sets up the LLM client and creates directories for storing mining results.
        Each run will be saved in a separate directory with a unique timestamp.
        
        Args:
            config: Configuration object for the miner
        """
        super(Miner, self).__init__(config=config)
        
        # Initialize the LLM client
        # You can override this in your config by setting model_name
        # Ensure we have a valid model name, defaulting to gpt-3.5-turbo if not specified
        self.model_name = os.getenv("CHUTES_MODEL_NAME", getattr(self.config.neuron, 'model_name', None) or 'deepseek-ai/DeepSeek-R1')
        if self.model_name is None:
            #self.model_name = 'llama3.2:1b'
            self.model_name = 'gpt-3.5-turbo'
            bt.logging.info(f"No model specified in config, using default model: {self.model_name}")
        
        bt.logging.info(f"Using LLM model: {self.model_name}")

        self.chutes_token = os.getenv("CHUTES_API_KEY")
        if not self.chutes_token:
            bt.logging.warning("CHUTES_API_KEY environment variable not set")
        
        self.api_base_url = os.getenv("CHUTES_API_BASE_URL", "https://llm.chutes.ai/v1")
        bt.logging.info(f"Using LLM API base URL: {self.api_base_url}")

        # Create a directory for storing mining results
        # This helps with debugging and analysis
        self.output_path = os.path.join(self.config.logging.logging_dir, "mining_results")
        os.makedirs(self.output_path, exist_ok=True)
        bt.logging.info(f"Mining results will be saved to: {self.output_path}")

        # batches can lead to truncated responses, so requests are split into
        # manageable chunks. The default batch size is tuned for DeepSeek and
        # can be overridden with the NAMES_PER_BATCH environment variable.
        self.names_per_batch = int(os.getenv("NAMES_PER_BATCH", 8))

        # Number of variations to generate for each name. Can be overridden
        # with the VARIATION_COUNT environment variable or the
        # --neuron.variation_count CLI argument.
        self.variation_count = int(
            os.getenv(
                "VARIATION_COUNT",
                getattr(self.config.neuron, "variation_count", 13),
            )
        )

    def build_prompt(self, names: List[str], variation_count: int = 13) -> str:
        """Create a concise prompt requesting name variations with clear format and a precise example."""
        bullet_points = [
            f"Provide {variation_count} variants for each name",
            "About half should sound similar and half should look similar",
            "Use a mix of vowel swap, consonant drop, and extra letter",
            "Output format: name:var1,var2,...",
            "Return one line per name with no extra text"
        ]
        instructions = " ".join(bp.rstrip('.').rstrip() + '.' for bp in bullet_points)
        names_text = ", ".join(names)
        example = (
            "Example:\n"
            "alexander:aleksandr,alexandre,alexandar,alexandor,alexender,alixander,alexandyr,alecksander,alexandr,alexandru,alexandarh,alejander,alexannder\n"
            "sophia:sophiaa,sofia,soffia,sophiya,sophya,sofea,soffiya,soophia,sofiya,sophiah,sofiah,sophina,sopheeah"
        )
        return f"For these names: {names_text}\n{instructions}\n{example}"

    async def forward(self, synapse: IdentitySynapse) -> IdentitySynapse:
        """
        Process a name variation request by generating variations for all names in a single batch.
        
        This is the main entry point for the miner's functionality. It:
        1. Receives a request with names and a query template
        2. Processes all names in a single LLM call for efficiency
        3. Extracts variations from the LLM response
        4. Returns the variations to the validator
        
        Each run is assigned a unique timestamp ID and results are saved in a
        dedicated directory for that run.
        
        Args:
            synapse: The IdentitySynapse containing names and query template
            
        Returns:
            The synapse with variations field populated with name variations
        """
        # Log incoming request details
        bt.logging.info(f"🎯 RECEIVED REQUEST from {synapse.dendrite.hotkey if synapse.dendrite else 'unknown'}")
        bt.logging.info(f"📝 Request contains {len(synapse.names)} names: {synapse.names[:3]}{'...' if len(synapse.names) > 3 else ''}")
        
        # Generate a unique run ID using timestamp
        run_id = int(time.time())
        bt.logging.info(f"Starting run {run_id} for {len(synapse.names)} names")
        
        # Get timeout from synapse (default to 120s if not specified)
        timeout = getattr(synapse, 'timeout', 120.0)
        bt.logging.info(f"Request timeout: {timeout:.1f}s for {len(synapse.names)} names")
        start_time = time.time()
        
        # Create a run-specific directory
        run_dir = os.path.join(self.output_path, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Check if we have enough time to process
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining < 5.0:  # Need at least 5 seconds for processing
            bt.logging.error("Insufficient time for processing")
            synapse.variations = {}
            return synapse
        
        try:
            bt.logging.info(
                f"Generating variations in batches of {self.names_per_batch} names"
            )
            variations = await self.process_name_batches_with_fallback(synapse.names, run_id, run_dir)
            synapse.variations = variations
        except Exception as e:
            bt.logging.error(f"Error in batch processing: {str(e)}")
            synapse.variations = {}
        
        # Log final timing information
        total_time = time.time() - start_time
        bt.logging.info(
            f"Request completed in {total_time:.2f}s of {timeout:.1f}s allowed. "
            f"Processed {len(synapse.names)} names in batch."
        )
        
        bt.logging.debug(f"======== SYNAPSE VARIATIONS===============================================: {synapse.variations}")
        bt.logging.debug(f"==========================Processed variations for {len(synapse.variations)} names in run {run_id}")
        bt.logging.debug(f"==========================Synapse: {synapse}")
        bt.logging.debug("End of request")
        return synapse


    async def Get_Respond_LLM(self, prompt: str, retries: int = 3) -> str:
        """
        Query the LLM using the Chutes API.

        This function sends a prompt to the remote LLM and returns its response.
        """
        # Include a system prompt to enforce concise, formatted output
        system_prompt = (
            "Generate alternative spellings only. "
            "Respond with lines formatted as 'name:var1,var2,...' and nothing else."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1500,
            "temperature": 0.7,
        }

        api_url = f"{self.api_base_url}/chat/completions"
        backoff = 1
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.chutes_token}",
                            "Content-Type": "application/json",
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as r:
                        if r.status != 200:
                            raise Exception(f"API returned status {r.status}: {await r.text()}")

                        data = await r.json()

                        if "choices" not in data or not data["choices"]:
                            raise Exception("Invalid response structure")

                        content = data["choices"][0]["message"]["content"].strip()
                        if not content:
                            raise Exception("Empty response from LLM")

                        return content
            except Exception as e:
                if attempt == retries - 1:
                    bt.logging.error(f"LLM query failed after {retries} attempts: {str(e)}")
                    raise
                bt.logging.warning(f"LLM request failed ({e}), retrying in {backoff}s")
                await asyncio.sleep(backoff)
                backoff *= 2


    def process_variations(self, Response_list: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """
        Process LLM responses to extract name variations.
        
        This function takes the raw LLM responses and extracts the name variations
        using the Process_function. It handles the parsing and cleaning of the
        LLM outputs, ensuring that all variations are properly cleaned before
        being returned or saved.
        
        Args:
            Response_list: List of LLM responses in the format:
                          ["Respond", "---", "Query-{name}", "---", "{LLM response}"]
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            
        Returns:
            Dictionary mapping each name to its list of variations
        """
        bt.logging.info(f"Processing {len(Response_list)} responses")
        # Split the responses by "Respond" to get individual responses
        Responds = "".join(Response_list).split("Respond")
        
        # Create a dictionary to store each name and its variations
        name_variations = {}
        
        # Process each response to extract variations
        for i in range(1, len(Responds)):
            try:
                # Process the response to extract the name and variations
                # Returns: (seed_name, processing_method, variations_list)
                llm_respond = self.Process_function(Responds[i], False)
                
                # Extract the seed name and variations
                name = llm_respond[0]
                
                # Filter out empty or NaN variations
                variations = [var for var in llm_respond[2] if not pd.isna(var) and var != ""]
                
                # Clean and deduplicate variations
                cleaned_variations = []
                for var in variations:
                    cleaned_var = var.replace(")", "").replace("(", "").replace("]", "").replace("[", "").replace(",", "").strip()
                    if cleaned_var:
                        cleaned_variations.append(cleaned_var)

                cleaned_variations = self.deduplicate_and_limit(name, cleaned_variations)
                name_variations[name] = cleaned_variations
                bt.logging.debug(f"Name variations: {name_variations}")

                bt.logging.debug(f"Processed {len(cleaned_variations)} variations for {name}")
            except Exception as e:
                bt.logging.error(f"Error processing response {i}: {e}")
        
        # # Save processed variations to JSON for debugging and analysis
        # self.save_variations_to_json(name_variations, run_id, run_dir)

        return name_variations

    def deduplicate_and_limit(self, seed: str, variations: List[str], count: Optional[int] = None) -> List[str]:
        """Remove duplicates and enforce a fixed number of variations."""
        if count is None:
            count = self.variation_count

        seen = set()
        cleaned = []
        for var in variations:
            v = var.strip()
            if not v or v.lower() == seed.lower():
                continue
            if v.lower() not in seen:
                seen.add(v.lower())
                cleaned.append(v)
            if len(cleaned) == count:
                return cleaned

        # Deterministic fallbacks if not enough variations
        extras = [seed + "e", seed + "y", seed[:-1] if len(seed) > 3 else seed]
        for extra in extras:
            if len(cleaned) >= count:
                break
            v = extra.strip()
            if v.lower() not in seen:
                seen.add(v.lower())
                cleaned.append(v)

        return cleaned[:count]
    
    def process_batch_response(self, batch_response: str, original_names: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """
        Process a batch LLM response to extract name variations for all names at once.
        
        This function takes a single LLM response containing variations for multiple names
        and extracts the variations for each name. The preferred format is one line per
        name in the form:
        ``name:var1,var2,...``
        
        Args:
            batch_response: The LLM response containing variations for all names
            original_names: List of original names that were sent to the LLM
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            
        Returns:
            Dictionary mapping each name to its list of variations
        """
        bt.logging.info(f"Processing batch response for {len(original_names)} names")

        # Create a dictionary to store each name and its variations
        name_variations = {}

        try:
            # Clean the response
            cleaned_response = batch_response.strip()

            # Split response on newlines or semicolons
            lines = re.split(r"[;\n]+", cleaned_response)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if ":" not in line or line.count(",") < 2:
                    bt.logging.warning(f"Ignoring malformed line: {line}")
                    continue
                name_part, variations_part = line.split(":", 1)
                name = name_part.strip()
                var_tokens = [v.strip() for v in variations_part.split(',') if v.strip()]
                if name and var_tokens:
                    cleaned = self.deduplicate_and_limit(name, var_tokens)
                    name_variations[name] = cleaned
                    bt.logging.info(f"Extracted {len(cleaned)} variations for {name}")

            # If the regex did not produce results, try alternative parsing
            if not name_variations:
                bt.logging.warning("Expected format not found, trying alternative parsing")
                
                # Try to match names with their variations using different patterns
                for name in original_names:
                    variations = []
                    
                    # Look for variations near the name in the response
                    name_patterns = [
                        f"{name}:",
                        f"{name} -",
                        f"{name}=",
                        f"{name} "
                    ]
                    
                    for pattern in name_patterns:
                        if pattern in cleaned_response:
                            # Extract text after the pattern
                            start_idx = cleaned_response.find(pattern) + len(pattern)
                            end_idx = cleaned_response.find(";", start_idx)
                            if end_idx == -1:
                                end_idx = len(cleaned_response)
                            
                            variations_text = cleaned_response[start_idx:end_idx].strip()
                            
                            # Split by comma and clean
                            for var in variations_text.split(","):
                                cleaned_var = var.strip()
                                if cleaned_var and cleaned_var not in variations:
                                    variations.append(cleaned_var)
                            
                            if variations:
                                break
                    
                    if variations:
                        cleaned = self.deduplicate_and_limit(name, variations)
                        name_variations[name] = cleaned
                        bt.logging.info(f"Extracted {len(cleaned)} variations for {name}")
            
            # If still no variations found, create a fallback response
            if not name_variations:
                bt.logging.warning("No variations extracted, creating fallback response")
                for name in original_names:
                    # Create simple variations based on common patterns
                    variations = []
                    name_lower = name.lower()
                    
                    # Add some basic variations
                    if len(name) > 2:
                        variations.append(name + "e")
                        variations.append(name + "y")
                        if name_lower.endswith('e'):
                            variations.append(name[:-1] + "ey")
                        if name_lower.endswith('y'):
                            variations.append(name[:-1] + "ie")
                    
                    if variations:
                        cleaned = self.deduplicate_and_limit(name, variations)
                        name_variations[name] = cleaned
                        bt.logging.info(f"Created {len(cleaned)} fallback variations for {name}")
            
            bt.logging.info(f"Successfully processed variations for {len(name_variations)} names")
            
        except Exception as e:
            bt.logging.error(f"Error processing batch response: {e}")
            # Create minimal fallback variations
            for name in original_names:
                fallback_vars = [name + "e", name + "y"]
                name_variations[name] = self.deduplicate_and_limit(name, fallback_vars)
        
        return name_variations

    async def fallback_individual_processing(self, names: List[str], existing_variations: Dict[str, List[str]], remaining_time: float, run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """
        Fallback method to process names individually when batch processing fails or is incomplete.
        
        Args:
            names: List of names to process
            existing_variations: Already processed variations
            remaining_time: Time remaining for processing
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
            
        Returns:
            Dictionary mapping each name to its list of variations
        """
        bt.logging.info(f"Starting fallback individual processing for {len(names)} names")
        
        # Copy existing variations
        variations = existing_variations.copy()
        
        # Calculate time per name (reserve 10% for processing overhead)
        available_time = remaining_time * 0.9
        time_per_name = available_time / len(names) if names else 0
        
        start_time = time.time()
        
        for i, name in enumerate(names):
            # Skip if already processed
            if name in variations:
                continue
            
            # Check if we have enough time
            elapsed = time.time() - start_time
            if elapsed > available_time:
                bt.logging.warning(f"Time limit reached during fallback processing, processed {i}/{len(names)} names")
                break
            
            try:
                # Simple prompt for individual processing
                simple_prompt = (
                    f"Generate 5-8 comma-separated alternative spellings for {name}. "
                    "Respond only with the names"
                )
                
                # Query LLM with shorter timeout
                response = await self.Get_Respond_LLM(simple_prompt)
                
                # Extract variations from response
                response_variations = []
                for var in response.split(","):
                    cleaned_var = var.strip()
                    if cleaned_var and cleaned_var != name and cleaned_var not in response_variations:
                        response_variations.append(cleaned_var)
                
                if response_variations:
                    cleaned = self.deduplicate_and_limit(name, response_variations)
                    variations[name] = cleaned
                    bt.logging.info(f"Fallback processed {name}: {len(cleaned)} variations")
                else:
                    # Create basic fallback variations
                    basic = [name + "e", name + "y"]
                    variations[name] = self.deduplicate_and_limit(name, basic)
                    bt.logging.info(f"Created basic fallback variations for {name}")
                
            except Exception as e:
                bt.logging.error(f"Error processing {name} in fallback: {str(e)}")
                # Create basic fallback variations
                basic = [name + "e", name + "y"]
                variations[name] = self.deduplicate_and_limit(name, basic)
        
        return variations

    async def process_single_batch(self, batch_names: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """Query the LLM for a batch of names and parse the response."""
        prompt = self.build_prompt(batch_names, self.variation_count)
        response = await self.Get_Respond_LLM(prompt)
        variations = self.process_batch_response(response, batch_names, run_id, run_dir)
        for name in batch_names:
            if name not in variations:
                variations[name] = self.deduplicate_and_limit(name, [])
        return variations

    async def process_name_batches(self, names: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """Process the entire list of names in parallel batches."""
        tasks = []
        batches = []
        for i in range(0, len(names), self.names_per_batch):
            batch = names[i : i + self.names_per_batch]
            batches.append(batch)
            tasks.append(self.process_single_batch(batch, run_id, run_dir))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        variations: Dict[str, List[str]] = {}
        for batch, result in zip(batches, results):
            if isinstance(result, Exception):
                bt.logging.error(f"Batch failed: {result}")
                for name in batch:
                    variations[name] = self.deduplicate_and_limit(name, [])
            else:
                variations.update(result)

        return variations

    async def process_name_batches_with_fallback(self, names: List[str], run_id: int, run_dir: str) -> Dict[str, List[str]]:
        """Process names in batches, retrying individually if batch fails or yields too few variations."""
        tasks = []
        batches = []
        for i in range(0, len(names), self.names_per_batch):
            batch = names[i : i + self.names_per_batch]
            batches.append(batch)
            tasks.append(self.process_single_batch_with_raw(batch, run_id, run_dir))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        variations: Dict[str, List[str]] = {}
        for batch, (result, raw_response) in zip(batches, results):
            if isinstance(result, Exception):
                bt.logging.error(f"Batch failed: {result}")
                for name in batch:
                    variations[name] = await self.individual_with_fallback(name, run_id, run_dir)
            else:
                for name in batch:
                    vars_for_name = result.get(name, [])
                    if len(vars_for_name) < 13:
                        bt.logging.warning(f"Name '{name}' got only {len(vars_for_name)} variations in batch, retrying individually.")
                        variations[name] = await self.individual_with_fallback(name, run_id, run_dir)
                    else:
                        variations[name] = vars_for_name
            # Log raw LLM response for debugging
            bt.logging.debug(f"Raw LLM batch response: {raw_response}")
        return variations

    async def process_single_batch_with_raw(self, batch_names: List[str], run_id: int, run_dir: str):
        """Like process_single_batch, but also returns the raw LLM response for logging."""
        prompt = self.build_prompt(batch_names)
        response = await self.Get_Respond_LLM(prompt)
        variations = self.process_batch_response(response, batch_names, run_id, run_dir)
        for name in batch_names:
            if name not in variations:
                variations[name] = []
        return (variations, response)

    async def individual_with_fallback(self, name: str, run_id: int, run_dir: str) -> List[str]:
        """Try to get variations for a single name, fallback to deterministic if LLM fails."""
        try:
            prompt = self.build_prompt([name])
            response = await self.Get_Respond_LLM(prompt)
            vars_dict = self.process_batch_response(response, [name], run_id, run_dir)
            vars_for_name = vars_dict.get(name, [])
            if len(vars_for_name) < 13:
                bt.logging.warning(f"Name '{name}' got only {len(vars_for_name)} variations individually, using fallback.")
                return self.deduplicate_and_limit(name, vars_for_name)
            return vars_for_name
        except Exception as e:
            bt.logging.error(f"Individual LLM call failed for '{name}': {e}")
            return self.deduplicate_and_limit(name, [])

    def save_variations_to_json(self, name_variations: Dict[str, List[str]], run_id: int, run_dir: str) -> None:
        """
        Save processed variations to JSON and DataFrame for debugging and analysis.
        
        This function saves the processed variations in multiple formats:
        1. A pandas DataFrame saved as a pickle file in the run-specific directory
        2. A JSON file with the name variations in the run-specific directory
        3. A JSON file with the model name and run ID in the main output directory
        
        Each file is named with the run ID to distinguish between different runs.
        
        Args:
            name_variations: Dictionary mapping names to variations
            run_id: Unique identifier for this processing run
            run_dir: Directory to save run-specific files
        """
        bt.logging.debug(f"Name variations: {name_variations}")
        bt.logging.debug(f"Run ID: {run_id}")
        bt.logging.debug(f"Run directory: {run_dir}")
        bt.logging.info("Saving variations to JSON and DataFrame")

        # Find the maximum number of variations for any name
        max_variations = max([len(vars) for vars in name_variations.values()]) if name_variations else 0
        bt.logging.info(f"Maximum number of variations found: {max_variations}")
        
        # Create a DataFrame with columns for the name and each variation
        columns = ['Name'] + [f'Var_{i+1}' for i in range(max_variations)]
        result_df = pd.DataFrame(columns=columns)
        
        # Fill the DataFrame with names and their variations, padding with empty strings if needed
        for i, (name, variations) in enumerate(name_variations.items()):
            row_data = [name] + variations + [''] * (max_variations - len(variations))
            result_df.loc[i] = row_data
        
        # Note: We no longer need to clean the data here since it's already cleaned
        # in the process_variations function
        
        # Save DataFrame to pickle for backup and analysis
        # Include run_id in the filename
        #df_path = os.path.join(run_dir, f"variations_df_{run_id}.pkl")
        #result_df.to_pickle(df_path)
        
        # Convert DataFrame to JSON format
        json_data = {}
        for i, row in result_df.iterrows():
            name = row['Name']
            # Extract non-empty variations
            variations = [var for var in row[1:] if var != ""]
            json_data[name] = variations
        
        # Save to JSON file
        # Include run_id in the filename
        # json_path = os.path.join(run_dir, f"variations_{run_id}.json")
        # import json
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, indent=4)
        # bt.logging.info(f"Saved variations to: {json_path}")
        # bt.logging.info(f"DataFrame shape: {result_df.shape} with {max_variations} variation columns")
    
    def Clean_extra(self, payload: str, comma: bool, line: bool, space: bool, preserve_name_spaces: bool = False) -> str:
        """
        Clean the LLM output by removing unwanted characters.
        
        Args:
            payload: The text to clean
            comma: Whether to remove commas
            line: Whether to remove newlines
            space: Whether to remove spaces
            preserve_name_spaces: Whether to preserve spaces between names (for multi-part names)
        """
        # Remove punctuation and quotes
        payload = payload.replace(".", "")
        payload = payload.replace('"', "")
        payload = payload.replace("'", "")
        payload = payload.replace("-", "")
        payload = payload.replace("and ", "")
        
        # Handle spaces based on preservation flag
        if space:
            if preserve_name_spaces:
                # Replace multiple spaces with single space
                while "  " in payload:
                    payload = payload.replace("  ", " ")
            else:
                # Original behavior - remove all spaces
                payload = payload.replace(" ", "")
        
        if comma:
            payload = payload.replace(",", "")
        if line:
            payload = payload.replace("\\n", "")
        
        return payload.strip()

    def validate_variation(self, name: str, seed: str, is_multipart_name: bool) -> str:
        """
        Helper function to validate if a variation matches the seed name structure.
        
        Args:
            name: The variation to validate
            seed: The original seed name
            is_multipart_name: Whether the seed is a multi-part name
            
        Returns:
            str: The validated and cleaned variation, or np.nan if invalid
        """
        name = name.strip()
        if not name or name.isspace():
            return np.nan
        
        # Handle cases with colons (e.g., "Here are variations: Name")
        if ":" in name:
            name = name.split(":")[-1].strip()
        
        # Check length reasonability (variation shouldn't be more than 2x the seed length)
        if len(name) > 2 * len(seed):
            return np.nan
        
        # Check structure consistency with seed name
        name_parts = name.split()
        if is_multipart_name:
            # For multi-part seed names (e.g., "John Smith"), variations must also have multiple parts
            if len(name_parts) < 2:
                bt.logging.warning(f"Skipping single-part variation '{name}' for multi-part seed '{seed}'")
                return np.nan
        else:
            # For single-part seed names (e.g., "John"), variations must be single part
            if len(name_parts) > 1:
                bt.logging.warning(f"Skipping multi-part variation '{name}' for single-part seed '{seed}'")
                return np.nan
            
        return name

    def Process_function(self, string: str, debug: bool) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Process the LLM response to extract the seed name and variations.
        
        This function parses the LLM response to extract:
        1. The original seed name
        2. The list of name variations
        
        It handles different response formats from LLMs:
        - Comma-separated lists (preferred format)
        - Line-separated lists
        - Space-separated lists with numbering
        
        The function ensures variations match the structure of the seed name:
        - Single-part seed names (e.g., "John") only get single-part variations
        - Multi-part seed names (e.g., "John Smith") only get multi-part variations
        
        Args:
            string: The LLM response in the format:
                   "---\nQuery-{name}\n---\n{response}"
            debug: Whether to return debug information
            
        Returns:
            Tuple containing:
            - seed_name: The original name
            - processing_method: The method used to process the response (r1, r2, or r3)
            - variations_list: The list of extracted variations
            - payload: (if debug=True) The processed payload
        """
        # Split the response by "---" to extract the query and response parts
        splits = string.split('---')
        
        # Extract and analyze the seed name structure
        seed = splits[1].split("-")[1].replace(".", "").replace(",", "").replace("'", "")
        seed_parts = seed.split()
        is_multipart_name = len(seed_parts) > 1
        seed = self.Clean_extra(seed, True, True, True, preserve_name_spaces=is_multipart_name)
        
        bt.logging.info(f"Processing seed name: '{seed}' (multipart: {is_multipart_name})")
        
        # Extract the response payload
        payload = splits[-1]
        
        # Case 1: Comma-separated list (preferred format)
        if len(payload.split(",")) > 3:  # Check if we have at least 3 commas
            # Clean the payload but keep commas for splitting
            payload = self.Clean_extra(payload, False, True, True, preserve_name_spaces=is_multipart_name)
            
            # Remove numbering prefixes
            for num in range(10):
                payload = payload.replace(str(num), "")
            
            # Split by comma and process each variation
            variations = []
            for name in payload.split(","):
                cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                if not pd.isna(cleaned_var):
                    variations.append(cleaned_var)
            
            if debug:
                return seed, "r1", variations, payload
            return seed, "r1", variations
        
        # Case 2 & 3: Non-comma separated formats
        else:
            # Case 2: Line-separated list
            len_ans = len(payload.split("\\n"))
            if len_ans > 2:  # Multiple lines indicate line-separated format
                # Clean the payload but preserve newlines for splitting
                payload = self.Clean_extra(payload, True, False, True, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                # Process line-separated variations
                variations = []
                for name in payload.split("\\n"):
                    cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                    if not pd.isna(cleaned_var):
                        variations.append(cleaned_var)
            
                if debug:
                    return seed, "r2", variations, payload
                return seed, "r2", variations
            
            # Case 3: Space-separated list
            else:
                # Clean the payload but preserve spaces for multi-part names
                payload = self.Clean_extra(payload, True, True, False, preserve_name_spaces=is_multipart_name)
                
                # Remove numbering prefixes
                for num in range(10):
                    payload = payload.replace(str(num), "")
                
                variations = []
                if is_multipart_name:
                    # For multi-part names, we need to carefully group the parts
                    current_variation = []
                    parts = payload.split()
                    
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        if ":" in part:  # New variation starts after colon
                            if current_variation:
                                # Process completed variation
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                            current_variation = [part.split(":")[-1].strip()]
                        else:
                            current_variation.append(part)
                            # Check if we have collected enough parts for a complete name
                            if len(current_variation) == len(seed_parts):
                                cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                                if not pd.isna(cleaned_var):
                                    variations.append(cleaned_var)
                                current_variation = []
                
                    # Handle any remaining parts
                    if current_variation:
                        cleaned_var = self.validate_variation(" ".join(current_variation), seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                else:
                    # For single-part names, simple space splitting is sufficient
                    for name in payload.split():
                        cleaned_var = self.validate_variation(name, seed, is_multipart_name)
                        if not pd.isna(cleaned_var):
                            variations.append(cleaned_var)
                
                if debug:
                    return seed, "r3", variations, payload
                return seed, "r3", variations

    async def blacklist(
        self, synapse: IdentitySynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        
        This function implements security checks to ensure that only authorized
        validators can query this miner. It verifies:
        1. Whether the request has a valid dendrite and hotkey
        2. Whether the hotkey is registered in the metagraph
        3. Whether the hotkey has validator permissions (if required)
        
        Args:
            synapse: A IdentitySynapse object constructed from the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: Whether the request should be blacklisted
                - str: The reason for the decision
        """
        # Log all incoming requests for debugging
        bt.logging.info(f"🔍 BLACKLIST CHECK for request from {synapse.dendrite.hotkey if synapse.dendrite else 'unknown'}")
        
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # Get the UID of the sender
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        # Check if the hotkey is registered in the metagraph
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Check if the hotkey has validator permissions (if required)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: IdentitySynapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        
        This function assigns a priority to each request based on the stake of the
        calling entity. Requests with higher priority are processed first, which
        ensures that validators with more stake get faster responses.
        
        Args:
            synapse: The IdentitySynapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.
                  Higher values indicate higher priority.
        """
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # Get the UID of the caller
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        
        # Use the stake as the priority
        # Higher stake = higher priority
        priority = float(
            self.metagraph.S[caller_uid]
        )
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"----------------------------------Name Variation Miner running... {time.time()}")
            time.sleep(30)
