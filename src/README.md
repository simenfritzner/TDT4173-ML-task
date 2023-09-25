Example workflow:
When in main:
- git pull
- git branch -b [branch name] (Personal name or algorithm or something, remove -b if already made a branch)
yoyoyo
When in branch:
- At the start and end of each session:
- git pull origin main
- Handle merge conflicts
- Work work work on your computer
- Every now and then:
- git add .
- git commit -m "Explain short what you did"

When your done with your code and are sure it works:
- git pull origin main
- Handle merge conflicts
- git add .
- git commit -m "Your commit message here"
- git push origin [branch-name]
- Go to Pull requests on github and make the request
- Need to figure out how we do it with pull requests


