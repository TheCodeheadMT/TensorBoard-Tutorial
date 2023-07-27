# ensure updated system
#sudo apt update && sudo apt upgrade -y
#sudo snap refresh
#sudo snap install --classic code
# Running root, so no need for sudo
apt update && sudo apt upgrade -y
snap refresh
snap install --classic code
# install pip (python package manager) if not installed already
#sudo apt install -y python3-pip
apt install -y python3-pip
python3 -m pip install --upgrade pip
# install python requirements
python3 -m pip install -r imports.txt
#sudo mkdir logs
mkdir logs