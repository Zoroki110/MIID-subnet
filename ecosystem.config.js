module.exports = {
    apps: [
      {
        name: 'miid-miner',
        cwd: '/root/MIID-subnet',
        interpreter: '/root/MIID-subnet/miner_env/bin/python',
        script: 'neurons/miner.py',
        args: [
          '--netuid', '54',
          '--wallet.name', 'gaia',
          '--wallet.hotkey', 'sentinel-bt54',
          '--subtensor.network', 'finney',
          '--subtensor.chain_endpoint', 'wss://entrypoint-finney.opentensor.ai:443',
          '--logging.debug',
          '--axon.requires_validator_sign', 'true',
          '--neuron.model_name', 'deepseek-ai/DeepSeek-R1'
        ].join(' '),
        env: {
          CHUTES_API_KEY: 'cpk_68ac1700d4394eb785ae6141bca2fa0d.40ef91a6f4075c87a7f408b0bae61d60.fjcsejQd8DnnzCQ7EFG1z0bAcAm6UWpj',
          CHUTES_API_BASE_URL: 'https://llm.chutes.ai/v1',
          CHUTES_MODEL_NAME: 'deepseek-ai/DeepSeek-R1',
          MODEL_PROVIDER: 'chutes'
        }
      }
    ]
  }
  