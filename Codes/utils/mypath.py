import os

db_root = '/nfs/turbo/coe-hunseok/sshoouri/Pascal_dataset/'
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]

db_names = {'PASCALContext': 'PASCALContext'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)