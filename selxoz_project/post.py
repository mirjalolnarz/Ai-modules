import requests

url = "http://127.0.0.1:8000/api/predict/"
files = {'image': open(r'C:\Users\mirja\DjangoAPI\selxoz_project\v\Tomato___Bacterial_spot\0ab9c705-f29e-45ac-b786-9549b3c38f16___GCREC_Bact.Sp 3223.JPG', 'rb')}
data = {'algorithm': 'KNN'}

response = requests.post(url, files=files, data=data)
print(response.json())
