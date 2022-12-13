# step 0:
cd local project folder

# step 1:
git init 
-- create a .git folder in your project folder

# step 2:
git add . 
-- . means the whole project or ./test.txt means the specific file

#  Step 3
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

# step 4:
git commit -m "first commit"

# step 5:
git remote add origin git@github.com:user_name/repository.git 
-- copy address from github repository

# step 6:
git branch -M main

# step 7:
git push -u origin main

