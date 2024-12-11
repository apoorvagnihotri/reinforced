from litellm import completion

import rootutils
import litellm

root = rootutils.setup_root('.')

import httpx

# Create a custom SSL context that trusts your corporate certificates
custom_client = httpx.Client(
    verify=False,  # Disables SSL verification
    proxies={
        "http://": "http://browsercfg.edc.corpintra.net",
        "https://": "http://browsercfg.edc.corpintra.net"
    }
)

# Configure LiteLLM to use this client
litellm.client_session = custom_client


messages = [{ "content": "Hello, how are you?","role": "user"}]


custom_headers = {
    "X-Proxy-SSL-Verify": "false",
    # Any other headers you want to add
}

# openai call
response = completion(
    model="anthropic/claude-3-5-haiku-latest", 
    messages=messages,        
    # headers=custom_headers
    )

print(response)