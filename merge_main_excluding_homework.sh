#!/bin/bash

# Check if the user is inside a Git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "Error: Not inside a Git repository."
  exit 1
fi

# Fetch the latest changes from the main branch
echo "Fetching updates from the main branch..."
git fetch origin main

# Get the list of directories with "Homework" in their name
HOMEWORK_DIRS=$(find . -type d -name "*Homework*" | sed 's|^\./||')

if [ -n "$HOMEWORK_DIRS" ]; then
  echo "Found Homework directories:"
  echo "$HOMEWORK_DIRS"
else
  echo "No Homework directories found."
fi

# Stash the changes in Homework directories to prevent overwriting
echo "Stashing Homework directories..."
for dir in $HOMEWORK_DIRS; do
  git stash push -m "Stash for $dir" -- $dir
done

# Merge the main branch into the current branch
echo "Merging main branch into $(git branch --show-current)..."
git merge origin/main --no-ff -m "Merge main branch into $(git branch --show-current), excluding Homework directories"

# Restore the stashed Homework directories
echo "Restoring Homework directories..."
for dir in $HOMEWORK_DIRS; do
  STASH_ENTRY=$(git stash list | grep "$dir" | head -n 1 | awk '{print $1}')
  if [ -n "$STASH_ENTRY" ]; then
    git stash apply --index "$STASH_ENTRY"
    git stash drop "$STASH_ENTRY"
  fi
done

# Commit any restored changes automatically
if git status --porcelain | grep -q '^ M'; then
  echo "Committing restored changes in Homework directories..."
  git add .
  git commit -m "Preserve changes in Homework directories after merging main branch"
fi

echo "Merge complete. All changes have been committed."