#!/usr/bin/python
import random

# Generate events for 1000 timesteps
def generate_input_file(filename, num_timesteps=1000, num_events=500):
    with open(filename, 'w') as f:
        # Write the total number of timesteps at the top of the file
        f.write(f"{num_timesteps}\n")
        
        for _ in range(num_events):
            # Randomly choose the event type: 'source' or 'force'
            event_type = random.choice(['source', 'force'])

            # For 'source', generate a positive integer density
            if event_type == 'source':
                density = random.randint(1, 10)  # Generate an integer density between 1 and 10

            # For 'force', generate a vector along one of the positive axes (X, Y, or Z)
            elif event_type == 'force':
                axis = random.choice(['x', 'y', 'z'])
                if axis == 'x':
                    x, y, z = 1, 0, 0
                elif axis == 'y':
                    x, y, z = 0, 1, 0
                elif axis == 'z':
                    x, y, z = 0, 0, 1

            # Assign a random timestep between 0 and num_timesteps - 1
            timestep = random.randint(0, num_timesteps - 1)
            
            # Write the event to the file
            if event_type == 'source':
                f.write(f"{event_type} {density} {timestep}\n")
            elif event_type == 'force':
                f.write(f"{event_type} {x} {y} {z} {timestep}\n")

# Generate the input file with 1000 timesteps and 500 events
generate_input_file("events.txt", num_timesteps=100, num_events=20)
