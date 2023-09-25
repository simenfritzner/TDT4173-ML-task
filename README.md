Machine learning assigment in TDT4173. We want to predict the expected power generated from sun cells in three diffrent locations by using a forecast. We aim to train a model on years of data which include both meassured and predicted data as well as the the energy genetraded on the coressponding data.

Genereating an SSH-key:
- Open terminal
- To check if you got an SSH-key run next line in terminal:
- ls -al ~/.ssh
- If no file exist then you need to make a new one, can just press enter through all the pop-ups which comes after running code below (change email):
- ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
- Then run:
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_rsa

Adding SSH-key to Github:
- Windows: cat ~/.ssh/id_rsa.pub | clip
- MacOS: cat ~/.ssh/id_rsa.pub | pbcopy
- Go to GitHub -> Click on your profile picture in the top right corner -> Go to Settings -> In the sidebar, click on SSH and GPG keys -> Click New SSH key -> Give your key a title (e.g., "Personal Laptop") and paste the copied public key into the "Key" field -> Click Add SSH key.

Connecting to repo:
- Make sure to have git installed, otherwise install it
- Open terminal and navigate to desired folder
- git clone git@github.com:yadert/TDT4173-ML-task.git
