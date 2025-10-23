# 2025 Azure Dev Summit Demos

Demos for my session "Leveraging Small Language Models for Smarter Workflows" at [Azure Dev Summit 2025 in Lisbon](https://azuredevsummit.com/agenda/leveraging-small-language-models-for-smarter-workflows-0nzm/0s7abc4rfyk) (October 13-16, 2025).

### Prerequisites

Make sure to create an `.env` file in the root directory with the following variables:

```
AZURE_OPENAI_RESOURCE=
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=

AZURE_AI_PROJECT=
AZURE_AI_KEY=

### AZURE ML
SUBSCRIPTION_ID=
RESOURCE_GROUP=
WORKSPACE_NAME=
CLUSTER_NAME=

# Azure ML recommended defaults for fine-tuning
VM_SIZE=Standard_NC24ads_A100_v4
MIN_NODES=0
MAX_NODES=2
ENDPOINT_NAME=
DEPLOYMENT_NAME=
```