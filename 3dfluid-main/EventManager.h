#ifndef EVENT_MANAGER_H
#define EVENT_MANAGER_H

#include <string>
#include <vector>

// Define a structure to hold 3D vector data for force events
struct Vector3 {
  int x, y, z;
};

// Enum to define event types
enum EventType { ADD_SOURCE, APPLY_FORCE };

// Define a structure to represent an Event
struct Event {
  EventType type;
  int density;   // Used only for source events
  Vector3 force; // Used only for force events
  int timestep;  // Common for both source and force events

  // Constructor for source events
  Event(int d, int ts) : type(ADD_SOURCE), density(d), timestep(ts) {}

  // Constructor for force events
  Event(int fx, int fy, int fz, int ts)
      : type(APPLY_FORCE), force({fx, fy, fz}), timestep(ts) {}
};

// Class to manage a list of events
class EventManager {
private:
  std::vector<Event> events; // Vector to store events
  int total_timesteps;       // Total number of timesteps

public:
  EventManager() : total_timesteps(0) {}

  // Method to read events from a file and store them
  void read_events(const std::string &filename);

  // Method to get all events for a specific timestamp
  std::vector<Event> get_events_at_timestamp(int timestamp) const;

  // Get total timesteps
  int get_total_timesteps() const { return total_timesteps; }
};

#endif // EVENT_MANAGER_H
