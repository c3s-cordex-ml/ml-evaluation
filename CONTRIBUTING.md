# How to contribute code for the evaluation of ML-RCMs

To collaborate on code development

* You will need an account on GitHub.com

* Ideally you will link your local computer to Github.com, clone the repository and push changes to it.


## Link local computer and Github.com
* To Pull or Push between your local repository and Github.com, you will need to have set up a link e.g.
    - An ssh key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding
    - A personal access token YOUR-PERSONAL-ACCESS-TOKEN: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    - If using YOUR-PERSONAL-ACCESS-TOKEN, you will need to store this safely as it is used as a password when cloning repositories or pushing to GitHub e.g. replying to these commands:

Username: # your GitHub username

Password: # YOUR-PERSONAL-ACCESS-TOKEN


## Create local repository - clone
* Click the button "Code" (just to the left of the About section which is on the right of the screen).
* Copy the https web url 

* In your terminal used for coding, e.g. mobaxterm, write git clone with the copied web url:

**git clone https_web_url**

e.g.

**git clone https://github.com/c3s-cordex-ml/ml-evaluation.git**

Reply to these commands that follow

Username: # your GitHub username

Password: # YOUR-PERSONAL-ACCESS-TOKEN

**git config user.email "your.email@mail.com"**

**git config user.name "your github username"**


# How to add indices and performance diagnostics

**1. Raise an issue; 2. Pull requests or 3. Push direct code changes**

* You will need an account on GitHub.com


## 1. Raise an issue on GitHub.com

* Click on the Issues tab which has the image of a circle with a point in the centre.

    - This is on the row of menu options (Code - Issues - Pull requests - Actions ... - Settings).

    - Or alternatively, there is a button in the top right corner of the screen, just to the left of your user icon (for this route, make sure you select the correct repository to add an issue).

* Click the button for "New Issue"

* Input text for title and description e.g.
title: Request for Standardardised RMSE to be added as a performance diagnostic.
description: Any other useful details e.g.
    - Explanation of why this should be added and for which variables/indices...
    - Precise details for computation.
    - Display equation or example code.
    - Reference if appropriate.

* Click the button "create"


## 2 & 3. Pull requests or Push direct code changes

* These can be done by linking your local computer and Github.com and pushing changes.


## 2. Pull Requests

### Create a branch, make changes, then push to GitHub.com (from terminal) and submit Pull Request (from GitHub.com).

Go to local local directory where cloned GitHub repository.

**git branch** # list branches available and make sure in main (star denotes active branch)

**git branch new_branch_name** # create new branch with suitable name e.g. branch_add_sal_score

**git checkout new_branch_name** # switch to branch new_branch_name

Make updates to files.

**git add .**  # Add all files to next commit (or can choose specific files)

**git commit -m "Message summarising changes"**  # Commit your changes with a description in quotes

**git remote -v**   # Verifies the new remote URL

**git push --set-upstream origin new_branch_name**   # Creates a branch of same name on the remote repo and pushes your changes to this


Go to repository on GitHub.com

Follow steps to create a pull request:

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request

* On GitHub.com, navigate to the main page of the repository.

* In the "Branch" menu, choose the branch that contains your commits.

* Above the list of files, in the yellow banner, click Compare & pull request to create a pull request for the associated branch.

* Use the base branch drop down menu to select the branch you'd like to merge your changes into, then use the compare branch drop-down menu to choose the topic branch you made your changes in.

* Type a title and description for your pull request.

* To create a pull request that is ready for review, click Create Pull Request. 

OR

* To create a draft pull request, use the drop-down and select Create Draft Pull Request, then click Draft Pull Request. If you are the member of an organization, you may need to request access to draft pull requests from an organization owner.


### Make changes to code within a Pull Request, e.g. after someone has supplied review comments.

#### Directly in GitHub.com

Go to relevant branch, find file and edit directly, then commit.

#### Via local terminal

**git checkout new_branch_name** # Got to branch

**git pull** # make sure up to date wrt remote repo

MAKE CHANGES

**git add .** # Add all files to next commit (or can choose specific files)

**git commit -m "Message summarising changes"**  # Commit your changes with a description in quotes

**git push**



## 3. Push direct code changes

### Make changes directly on cloned repository then push to GitHub.com (from terminal)

Go to local local directory where cloned GitHub repository and make updates to files.

**git branch** # list branches available and make sure in main (star denotes active branch)

**git add .** # Add all files to next commit (or can choose specific files)

**git commit -m "Message summarising changes"**  # Commit your changes with a description in quotes

**git remote -v**   # Verifies the new remote URL

**git push**   # Push your changes



## More info on Git commands

https://education.github.com/git-cheat-sheet-education.pdf

e.g. 

**git status** # Shows branch, files that have been modified, files that have been added ready for the next commit

**git commit -a -m "Message summarising changes"**  # Adds AND Commits changes to files that are already associated with git (but doesn't add newly created files)

