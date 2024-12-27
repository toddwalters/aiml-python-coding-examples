from datetime import datetime, timedelta

class CarRental:
    """
    CarRental class represents a car rental service.
    
    Attributes:
    - inventory: An integer representing the number of cars available for rent.
    - rental_time: A datetime object representing the time when a car is rented.
    - rental_mode: A string representing the rental mode (Hourly, Daily, Weekly).
    - cars_rented: An integer representing the number of cars rented.
    
    Methods:
    - display_inventory(): Displays the current inventory of available cars.
    - rent_hourly(n): Rents n cars on an hourly basis.
    - rent_daily(n): Rents n cars on a daily basis.
    - rent_weekly(n): Rents n cars on a weekly basis.
    - return_car(rental_time, rental_mode, cars_rented): Returns the rented cars and calculates the bill.
    - print_object_variables(): Prints the object variables for debugging purposes.
    """
    
    enable_object_printing = False  # Flag variable to control object variable printing
    
    def __init__(self, inventory=0):
        """
        Initializes a CarRental object with the given inventory.
        
        Parameters:
        - inventory: An integer representing the number of cars available for rent (default: 0).
        """
        print('Welcome to the CarRental Class')
        self.inventory = inventory
        self.rental_time = None
        self.rental_mode = None
        self.cars_rented = 0
    
    def display_inventory(self):
        """
        Displays the current inventory of available cars.
        
        Returns:
        - The current inventory of available cars.
        """
        if CarRental.enable_object_printing:
            self.print_object_variables()
        print(f"We currently have {self.inventory} cars available to rent.")
        return self.inventory
    
    def rent_hourly(self, n):
        """
        Rents n cars on an hourly basis.
        
        Parameters:
        - n: An integer representing the number of cars to rent.
        
        Returns:
        - rental_time: The time when the cars were rented.
        - rental_mode: The rental mode (Hourly).
        - cars_rented: The number of cars rented.
        """
        if CarRental.enable_object_printing:
            self.print_object_variables()
        if n <= 0:
            print('Number of cars should be positive')
            return None, None, 0
        elif n > self.inventory:
            print(f"Sorry! We currently have {self.inventory} cars available to rent.")
            return None, None, 0
        elif self.cars_rented > 0:
            print(f"You have already rented a car. Please return it first.")
            return None, None, 0
        else:
            self.inventory -= n
            self.rental_time = datetime.now()
            self.rental_mode = 'Hourly'
            self.cars_rented = n
            if CarRental.enable_object_printing:
                self.print_object_variables()
            print(f"You have rented {n} car(s) on an hourly basis at {self.rental_time}.")
            return self.rental_time, self.rental_mode, self.cars_rented
    
    def rent_daily(self, n):
        """
        Rents n cars on a daily basis.
        
        Parameters:
        - n: An integer representing the number of cars to rent.
        
        Returns:
        - rental_time: The time when the cars were rented.
        - rental_mode: The rental mode (Daily).
        - cars_rented: The number of cars rented.
        """
        if CarRental.enable_object_printing:
            self.print_object_variables()
        if n <= 0:
            print('Number of cars should be positive')
            return None, None, 0
        elif n > self.inventory:
            print(f"Sorry! We currently have {self.inventory} cars available to rent.")
            return None, None, 0
        elif self.cars_rented > 0:
            print(f"You have already rented a car. Please return it first.")
            return None, None, 0
        else:
            self.inventory -= n
            self.rental_time = datetime.now()
            self.rental_mode = 'Daily'
            self.cars_rented = n
            if CarRental.enable_object_printing:
                self.print_object_variables()
            print(f"You have rented {n} car(s) on a daily basis at {self.rental_time}.")
            return self.rental_time, self.rental_mode, self.cars_rented
    
    def rent_weekly(self, n):
        """
        Rents n cars on a weekly basis.
        
        Parameters:
        - n: An integer representing the number of cars to rent.
        
        Returns:
        - rental_time: The time when the cars were rented.
        - rental_mode: The rental mode (Weekly).
        - cars_rented: The number of cars rented.
        """
        if CarRental.enable_object_printing:
            self.print_object_variables()
        if n <= 0:
            print('Number of cars should be positive')
            return None, None, 0
        elif n > self.inventory:
            print(f"Sorry! We currently have {self.inventory} cars available to rent.")
            return None, None, 0
        elif self.cars_rented > 0:
            print(f"You have already rented a car. Please return it first.")
            return None, None, 0
        else:
            self.inventory -= n
            self.rental_time = datetime.now()
            self.rental_mode = 'Weekly'
            self.cars_rented = n
            if CarRental.enable_object_printing:
                self.print_object_variables()
            print(f"You have rented {n} car(s) on a weekly basis at {self.rental_time}.")
            return self.rental_time, self.rental_mode, self.cars_rented
    
    def return_car(self, rental_time, rental_mode, cars_rented):
        """
        Returns the rented cars and calculates the bill.
        
        Parameters:
        - rental_time: The time when the cars were rented.
        - rental_mode: The rental mode (Hourly, Daily, Weekly).
        - cars_rented: The number of cars rented.
        
        Returns:
        - rental_time: The time when the cars were returned.
        - rental_mode: The rental mode (None).
        - cars_rented: The number of cars returned.
        """
        if rental_time is None or rental_mode is None or cars_rented <= 0:
            print("Invalid values submitted for return. Please provide valid values.")
            return None, None, None
        
        if rental_time and rental_mode and cars_rented:
            now = datetime.now()
            rental_period = now - rental_time
            
            if CarRental.enable_object_printing:
                self.print_object_variables()
            
            if rental_mode == 'Hourly':
                if rental_period.seconds < 3600:
                    rental_period += timedelta(seconds=(3600 - rental_period.seconds))
                    print("Minimum rental period for hourly rentals is 1 hour, rounding rental period for each car to 1 hour.")
                    if CarRental.enable_object_printing:
                        self.print_object_variables()
                if CarRental.enable_object_printing:
                    self.print_object_variables()
                bill = round(rental_period.seconds / 3600) * 5 * cars_rented
                print(f"Your bill is currently ${bill}")

            elif rental_mode == 'Daily':
                if rental_period.days < 1:
                    rental_period = timedelta(days=1)
                    print("Minimum rental period for daily rentals is 1 day, rounding rental period for each car to 1 day.")
                    if CarRental.enable_object_printing:
                        self.print_object_variables()
                if CarRental.enable_object_printing:
                    self.print_object_variables()
                bill = round(rental_period.days) * 20 * cars_rented
                print(f"Your bill is currently ${bill}")

            elif rental_mode == 'Weekly':
                if rental_period.days < 7:
                    rental_period = timedelta(days=7)
                    print("Minimum rental period for weekly rentals is 1 week, rounding rental period for each car to 1 week.")
                    if CarRental.enable_object_printing:
                        self.print_object_variables()
                if CarRental.enable_object_printing:
                    self.print_object_variables()
                bill = round(rental_period.days / 7) * 60 * cars_rented
                print(f"Your bill is currently ${bill}")
                
            if self.cars_rented >= 2:
                if CarRental.enable_object_printing:
                    self.print_object_variables()
                print("You have a 20% discount!")
                bill *= 0.8
                
            self.rental_time, self.rental_mode, self.cars_rented = None, None, 0
            self.inventory += cars_rented
            if CarRental.enable_object_printing:
                self.print_object_variables()
                
        self.rental_time, self.rental_mode, cars_rented = None, None, 0
        print(f"Thanks for returning your car(s). Your bill is ${bill}")
        return self.rental_time, self.rental_mode, cars_rented
    
    def print_object_variables(self):
        """
        Prints the object variables for debugging purposes.
        """
        print()
        for attr, value in self.__dict__.items():
            print(f"CarRental.{attr}: {value}")
        print()

class Customer:
    """
    Represents a customer in a car rental system.
    """
    def __init__(self):
        self.rental_mode = None
        self.rental_time = None
        self.cars_rented = 0

    def request_car(self, n):
        """
        Requests to rent a specified number of cars.
        Args:
            n (int): The number of cars to rent.
        Returns:
            int: The number of cars rented.
        """
        if n <= 0:
            print('Number of cars should be positive')
            return 0
        else:
            self.cars_rented = n
            return self.cars_rented

    def return_car(self):
        """
        Returns the rental time, rental mode, and number of cars rented.
        Returns:
            tuple: A tuple containing the rental time, rental mode, and number of cars rented.
        """
        if self.rental_mode and self.rental_time and self.cars_rented:
            return self.rental_time, self.rental_mode, self.cars_rented
        else:
            print("You have not rented any cars.")
            return None, None, 0