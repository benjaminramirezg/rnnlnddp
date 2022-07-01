cd ..
echo 'Creating virtual env...'
virtualenv torchenv
echo 'Activating virtual env...'
source torchenv/bin/activate
echo 'Installing dependencies'
python -m pip install -r requirements.txt
echo 'Deactivating virtual env'
deactivate