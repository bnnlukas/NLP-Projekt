import requests

url = "https://api-football-v1.p.rapidapi.com/v3/timezone"

headers = {
	"X-RapidAPI-Key": "5219b33050msh2d7c0f3dcf9a730p1f1924jsn170fedb9e97e",
	"X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers)

print(response.text)