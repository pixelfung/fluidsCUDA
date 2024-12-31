#include "EventManager.h"
#include <fstream>
#include <iostream>
#include <sstream>

// Method to read events from a file and store them
void EventManager::read_events(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  std::string line;

  // Read the first line to get the total number of timesteps
  std::getline(file, line);
  std::istringstream first_line(line);
  first_line >> total_timesteps;

  // Read the remaining lines for events
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string event_type;
    int x, y, z, density, timestep;

    iss >> event_type;

    if (event_type == "source") {
      // Parse the density and timestep for source event
      iss >> density >> timestep;
      events.emplace_back(density, timestep);
    } else if (event_type == "force") {
      // Parse the vector and timestep for force event
      iss >> x >> y >> z >> timestep;
      events.emplace_back(x, y, z, timestep);
    } else {
      std::cerr << "Unknown event type: " << event_type << std::endl;
    }
  }

  file.close();
}

// Method to get all events for a specific timestamp
std::vector<Event> EventManager::get_events_at_timestamp(int timestamp) const {
  std::vector<Event> result;
  for (const auto &event : events) {
    if (event.timestep == timestamp) {
      result.push_back(event);
    }
  }
  return result;
}
