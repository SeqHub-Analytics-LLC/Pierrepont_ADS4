""" **Requirements:**
1. Start with a fixed account balance, e.g., `$500`.
2. The program should present the user with the following options:
   - **1. Check Balance**: Display the current account balance.
   - **2. Deposit**: Prompt the user to enter an amount to deposit, then add it to the balance.
   - **3. Withdraw**: Prompt the user to enter an amount to withdraw. If the amount exceeds the current balance, display an error message.
   - **4. Exit**: Exit the program and print a goodbye message.
3. The program should run in a loop, displaying the menu after every action until the user selects the "Exit" option.

#### **Implementation Notes:**
- Validate inputs to ensure they are numeric where appropriate.

#### **Example Input/Output:**

Welcome to the ATM!
Your current balance is $500.

Select an option:
  1. Check Balance
  2. Deposit
  3. Withdraw
  4. Exit
Enter your choice: 1
Your current balance is $500.

Select an option:
  1. Check Balance
  2. Deposit
  3. Withdraw
  4. Exit
Enter your choice: 2
Enter deposit amount: 200
You have successfully deposited $200. Your new balance is $700.

Select an option:
  1. Check Balance
  2. Deposit
  3. Withdraw
  4. Exit
Enter your choice: 3
Enter withdrawal amount: 800
Error: Insufficient funds. Your balance is $700.

Select an option:
  1. Check Balance
  2. Deposit
  3. Withdraw
  4. Exit
Enter your choice: 4
Thank you for using the ATM. Goodbye!
"""

class ATM:
    def __init__(self, initial_balance=500):
        self.balance = initial_balance  

    def check_balance(self):
        print(f"Your current balance is ${self.balance}.")

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"You have successfully deposited ${amount}. Your new balance is ${self.balance}.")
        else:
            print("Invalid deposit amount. Please enter a positive number.")

    def withdraw(self, amount):
        if amount > 0:
            if amount <= self.balance:
                self.balance -= amount
                print(f"You have successfully withdrawn ${amount}. Your new balance is ${self.balance}.")
            else:
                print(f"Error: Insufficient funds. Your balance is ${self.balance}.")
        else:
            print("Invalid withdrawal amount. Please enter a positive number.")

    def run(self):
        print("Welcome to the ATM!")
        while True:
            print("\nSelect an option:")
            print("  1. Check Balance")
            print("  2. Deposit")
            print("  3. Withdraw")
            print("  4. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                self.check_balance()

            elif choice == "2":
                try:
                    amount = float(input("Enter deposit amount: "))
                    self.deposit(amount)
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

            elif choice == "3":
                try:
                    amount = float(input("Enter withdrawal amount: "))
                    self.withdraw(amount)
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

            elif choice == "4":
                print("Thank you for using the ATM. Goodbye!")
                break

            else:
                print("Invalid choice. Please select a valid option (1-4).")

atm = ATM()
atm.run()