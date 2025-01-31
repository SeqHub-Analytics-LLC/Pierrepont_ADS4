""" **Challenge 3: Employee Management System (Using Inheritance)**

#### **Objective**:
Create a program to manage employees in a company. There should be two types of employees:
1. **Regular Employee**:
   - Has attributes like `name`, `age`, and `salary`.
   - Can calculate their annual salary.
2. **Manager (Inherits from Employee)**:
   - Inherits all attributes from `Employee` and has an additional attribute `bonus`.
   - Overrides the method to calculate annual salary to include the bonus.

#### **Requirements**:
1. Create a base class `Employee` with attributes `name`, `age`, and `salary`.
   - Add a method `get_annual_salary()` that calculates the annual salary.
2. Create a derived class `Manager` that inherits from `Employee`.
   - Add an additional attribute `bonus`.
   - Override the `get_annual_salary()` method to include the bonus.
3. Test the classes by creating both regular employees and managers, and calculate their annual salaries.

---

#### **Example Input/Output**:

##### **Example 1**:
# Create regular employees
emp1 = Employee("Alice", 25, 50000)
emp2 = Employee("Bob", 30, 60000)

# Create managers
mgr1 = Manager("Charlie", 40, 80000, 10000)

# Calculate annual salaries
print(emp1.get_annual_salary())  # Output: 50000
print(emp2.get_annual_salary())  # Output: 60000
print(mgr1.get_annual_salary())  # Output: 90000 (80000 + 10000)
```
"""

## Your code here
class Employee:
    """Base class representing a regular employee."""
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

    def get_annual_salary(self):
        """Calculates and returns the annual salary."""
        return self.salary

class Manager(Employee):
    """Derived class representing a Manager, inheriting from Employee."""
    def __init__(self, name, age, salary, bonus):
        super().__init__(name, age, salary)  # Inherit from Employee
        self.bonus = bonus

    def get_annual_salary(self):
        """Overrides the method to include the bonus in the annual salary."""
        return self.salary + self.bonus
# Test cases
emp1 = Employee("Alice", 25, 50000)
emp2 = Employee("Bob", 30, 60000)
mgr1 = Manager("Charlie", 40, 80000, 10000)

print(emp1.get_annual_salary())  # Output: 50000
print(emp2.get_annual_salary())  # Output: 60000
print(mgr1.get_annual_salary())  # Output: 90000 (80000 + 10000)
