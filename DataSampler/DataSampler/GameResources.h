#pragma once
#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"
#include "inc\main.h"

#include "myTypes.h"
#include "myEnums.h"

#include <vector>
#include <functional>

extern std::vector<Vehicle> createdVehicles;
extern std::vector<Ped> createdDrivers;
extern std::vector<TaskSequence> createdCarSequences;
extern std::vector<Ped> createdPeds;
extern std::vector<TaskSequence> createdPedSequences;

Entity spawnAtCoords(LPCSTR modelName, int modelType, Vector3d coords, float heading = 0.0);
Ped spawnDriver(Vehicle vehicle, LPCSTR modelName);
TaskSequence createTaskSequence(Ped p, std::function<void()> actionItem);
void deleteAllCreated();
void deleteCar(int i);
void deletePed(int i);
