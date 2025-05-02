import pandas as pd

data = [
    {
        'id': '1',
        'user_id': '6a9169d9-2385-49fc-a24a-99b8eb9437e8',
        'field_name': 'F1',
        'field_size': '10x20',
        'location': '35.6895,139.6917',
        'notes': 'North field near river'
    },
    {
        'id': '2',
        'user_id': '6a9169d9-2385-49fc-a24a-99b8eb9437e8',
        'field_name': 'F2',
        'field_size': '15x25',
        'location': '35.6897,139.6920',
        'notes': 'South field with sandy soil'
    },
    {
        'id': '3',
        'user_id': '6a9169d9-2385-49fc-a24a-99b8eb9437e8',
        'field_name': 'F3',
        'field_size': '12x18',
        'location': '35.6900,139.6900',
        'notes': 'East field with irrigation system'
    }
]

df = pd.DataFrame(data)
df.to_excel('data/fields.xlsx', index=False)
print("fields.xlsx created successfully with sample data.")
