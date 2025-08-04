import requests

token = "7W2BDVtnhys7QYKiaMUVcg"

print('TESTE 1: AÇÕES')
ticker = "ITUB4"
url = f"https://brapi.dev/api/quote/{ticker}?token={token}&range=1d&interval=1m&fundamental=true&dividends=true&modules=summaryProfile%2CbalanceSheetHistory"


response = requests.get(
    url,
    headers={"Authorization": f"Bearer {token}"}
    )
if response.status_code == 200:
    data = response.json()
    print(data['results'])
else:
    print(f"Erro: {response.status_code}")

print('TESTE 2: CRIPTOS')
cripto = "BTC"
url = f"https://brapi.dev/api/v2/crypto/available?search={cripto}&token={token}"

response = requests.request("GET", url, headers = {
                            "Authorization": f"Bearer {token}"
                            })

print(response.text)