import orbital_vault as ov
from CustomerDatabase import CustomerDatabase
from pimsys.regions.RegionsDb import RegionsDb
from datetime import datetime, timedelta
import psycopg2 as sqlsystem


def get_utc_timestamp(x: datetime):
    return int((x - datetime(1970, 1, 1)).total_seconds())

#ov.configure_secrets_backend(use_secretsmanager=True)
config = ov.get_sarccdb_credentials()
#config['database'] = config.pop('dbname')
#customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])
#project = customer_db.get_project_by_name(customer)


start_date = '2025-02-10'
end_date = '2025-02-12'

# config['database'] = config.pop('dbname')
schema = 'optical_ingestion'
connection_string = f"host={config['host']} port={config['port']} user='{config['user']}' password='{config['password']}' dbname={config['database']} options='-c search_path={schema}'"
with sqlsystem.connect(connection_string) as db:
    # Get a cursor object
    cursor = db.cursor()
    # Search archive images
    cursor.execute("SELECT * FROM products WHERE ((%(start_date)s <= start_time AND start_time <= %(end_date)s) OR (%(start_date)s <= end_time AND end_time <= %(end_date)s)) and source = %(source)s",
                    {'start_date': get_utc_timestamp(start_date),
                    'end_date': get_utc_timestamp(end_date),
                    #bounds': sqlsystem.Binary(location.wkb),
                    'source': 'SkySat'})
    output = cursor.fetchall()



'''
customer = 'PTT-2024'
domain = 'ptt.orbitaleye.nl'
corridor_width = 1000  # meters

date_start = '2025-02-10'
date_end = '2025-02-12'
    
project_start = datetime.strptime(date_start, '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
project_end = datetime.strptime(date_end, '%Y-%m-%d') 

#ov.configure_secrets_backend(use_secretsmanager=True)
regions_db_cred = ov.get_sarccdb_credentials()
settings_db = {
                    "host": regions_db_cred.get("host"),
                    "port": regions_db_cred.get("port"),
                    "user": regions_db_cred.get("user"),
                    "password": regions_db_cred.get("password"),
                    "database": regions_db_cred.get("dbname"),
                    "schema": "tpi_dashboard,public",
                }

customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])

project = customer_db.get_project_by_name(customer)

print('ok')
with RegionsDb(settings_db) as database:
    print('ok2')
    database_customer = database.get_regions_by_customer(customer)
    print(database_customer)
'''
