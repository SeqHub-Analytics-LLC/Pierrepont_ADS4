# Applied Data Science 4 - Central Repository

Welcome, Pierrepont students, to the Applied Data Science 4 course repository! 🎉

This repository will serve as the central hub for all course-related scripts, sample projects, and resources. As we progress through the course, you’ll find folders in the repository’s home directory dedicated to subsequent course modules. This ensures that everything you need for coding and project work is organized and easily accessible.

While Google Classroom will remain active for general communication and announcements, it is not ideal for sharing scripts or script files. Therefore, we will be using GitHub as our primary platform for code sharing and collaboration.

# Getting Started with GitHub

If you’re new to GitHub or need a refresher, don’t worry! I’ve compiled a collection of GitHub resources on Google Drive to help you get started. These resources cover essential topics like setting up your account, using Git commands, and collaborating on repositories.

📥 Access the resources here: [Google Drive Link](https://drive.google.com/drive/folders/1mTdMxZIehI50AFTnFDoJwx5sK7SQu1Xw?usp=drive_link)

# Repository Setup

1. **Download Vscode**
   - Vscode is the recommended code editor for this course.
   - Download Vscode [here](https://code.visualstudio.com/)
   - Vscode Setup [Tutorial](https://youtu.be/zulGMYg0v6U?feature=shared)
     
2. **Clone the Repository**
   - Open a terminal or command prompt.
   - Navigate to the folder where you want to store the repository:   
  ```python
  cd /path/to/your/folder
  ```

3. **Clone the repository using the following command:**
  ```python
  git clone https://github.com/SeqHub-Analytics-LLC/Pierrepont_ADS4.git
  ```

4. **Navigate into the project folder:**
   Use the command below
  ```python
  cd Pierrepont_ADS4
  ```

# Setting Up a Virtual Environment

To ensure a clean and isolated Python environment, follow these steps:

1. **Create a Virtual Environment:**
   
  ```python
  python -m venv venv
  ```
2. **Activate the Virtual Environment:**
 - If you are a Windows user:

  ```python
 venv\Scripts\activate
  ```
 - If you are a Mac/Linux user:

  ```python
source venv/bin/activate
  ```

3. **Install Required Packages:**
   
  ```python
  pip install -r requirements.txt
  ```

4. **Configure VS Code for the Virtual Environment:**
 - Open the repository in VS Code:
  - Press ```Ctrl+Shift+P``` (or ```Cmd+Shift+P``` on Mac) to open the Command Palette.
  - **Select Python: Select Interpreter and choose the virtual environment**


# Understanding the Structure of this repository

The repository is organized into folders for each module in the course. New folders and files will be added as we progress. Here’s a quick preview:
 - module_1/: Scripts and projects for Module 1
 - module_2/: Scripts and projects for Module 2
 - ...: Additional modules will be added over time

# How to Pull Updates

**Pull Updates:**
   - Always pull the latest changes before starting work:
  ```python
  git pull origin main
  ```

# How to Push Updates

1. **Step 1: Initialize Git Repository**
  - Navigate to your project directory and initialize a Git repository (if not already done):
  ```bash
  git init
  ```

2. **Step 2: Add Files**
  - Stage all files for commit:
  ```bash
  git add .
  ```

3. **Step 3: Commit Changes**
  - Commit the staged files:

  ```bash
  git commit -m "commit message"
  ```
4. **Add Remote Repository**
  - Add your GitHub repository as the remote origin:
  ```bash
  git remote add origin https://github.com/USERNAME/Pierrepont_ADS4.git
  ```
  - Replace `USERNAME` with your GitHub username.


5. **Creating your personal Branch**
Create and switch to a new branch, Use your name:
```bash
git checkout -b StudentName
```

6. **Push to your GitHub branch**
Push the branch to the GitHub repository:
```bash
git push -u origin StudentName
```

## Support and Resources

 - If you encounter issues with Git or GitHub, refer to the Google Drive resources or reach out to me for assistance.
 - For issues related to scripts or projects, raise them in the GitHub Issues tab or Google Classroom.

## Let’s Get Started!

I’m excited to see how you leverage this repository to learn, collaborate, and build amazing projects. Let’s make this a productive and enjoyable journey in Applied Data Science 4!

Happy coding! 🚀
- **Taiwo Togun**
