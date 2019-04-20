#pragma once
#include <functional>

#include "inc\natives.h"
#include "inc\types.h"
#include "inc\enums.h"
#include "inc\main.h"

#include "myEnums.h"
#include "myTypes.h"

namespace GameResources {
	extern std::vector<PedWithMission> createdPeds;
	extern std::vector<VehicleWithMission> createdVehicles;

	VehicleWithMission& spawnVehicleAtCoords(LPCSTR modelName, Vector3d coords, float heading = 0.0f);
	PedWithMission& spawnPedAtCoords(LPCSTR modelName, Vector3d coords, float heading = 0.0f);
	VehicleWithMission& spawnDriver(VehicleWithMission& vehicle, LPCSTR modelName);
	PedWithMission& createPedTask(PedWithMission& p, std::function<void()> actionItem);
	VehicleWithMission& createVehTask(VehicleWithMission& v, std::function<void()> actionItem);

	void deleteVeh(int i);
	void deletePed(int i);
	void deleteAllCreated();
};
