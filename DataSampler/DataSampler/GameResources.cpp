#include "GameResources.h"

std::vector<Vehicle> createdVehicles;
std::vector<Ped> createdDrivers;
std::vector<TaskSequence> createdCarSequences;
std::vector<Ped> createdPeds;
std::vector<TaskSequence> createdPedSequences;

Entity spawnAtCoords(LPCSTR modelName, int modelType, Vector3d coords, float heading) {
	DWORD model;
	if (modelName[0] != '\0') {
		model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
		STREAMING::REQUEST_MODEL(model);
		while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
	}
	Entity e;

	if (modelType == MODEL_VEH) {
		Vehicle veh;
		veh = VEHICLE::CREATE_VEHICLE(model, coords.x, coords.y, coords.z, heading, 1, 1);
		VEHICLE::SET_VEHICLE_ON_GROUND_PROPERLY(veh);
		createdVehicles.push_back(veh);
		e = veh;
	}
	else if (modelType == MODEL_PED) {
		Ped p;
		if (modelName[0] != '\0') {
			p = PED::CREATE_PED(PED_TYPE_MISSION, model, coords.x, coords.y, coords.z, heading, FALSE, FALSE);
		}
		else {
			p = PED::CREATE_RANDOM_PED(coords.x, coords.y, coords.z);
		}
		PED::SET_PED_MAX_HEALTH(p, 100);
		createdPeds.push_back(p);
		e = p;
	}

	if (modelName[0] != '\0') {
		STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
	}

	return e;
}

Ped spawnDriver(Vehicle vehicle, LPCSTR modelName) {
	Ped p;
	DWORD model;

	if (modelName[0] != '\0') {
		model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
		STREAMING::REQUEST_MODEL(model);
		while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
		p = PED::CREATE_PED_INSIDE_VEHICLE(vehicle, 4, model, -1, 1, 1);
	}
	else {
		p = PED::CREATE_RANDOM_PED_AS_DRIVER(vehicle, true);
	}
	PED::SET_PED_MAX_HEALTH(p, 100);
	createdDrivers.push_back(p);

	if (modelName[0] != '\0') {
		STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
	}

	return p;
}

TaskSequence createTaskSequence(Ped p, std::function<void()> actionItem) {
	TaskSequence sequence = 0;
	AI::OPEN_SEQUENCE_TASK(&sequence);
	actionItem();
	AI::CLOSE_SEQUENCE_TASK(sequence);
	AI::TASK_PERFORM_SEQUENCE(p, sequence);
	createdPedSequences.push_back(sequence);
	return sequence;
}

void deleteAllCreated() {
	for (int i = createdPedSequences.size(); i > 0; i--) {
		int sequence = createdPedSequences.back();
		AI::CLEAR_SEQUENCE_TASK(&sequence);
		createdPedSequences.pop_back();
	}

	for (int i = createdCarSequences.size(); i > 0; i--) {
		int sequence = createdCarSequences.back();
		AI::CLEAR_SEQUENCE_TASK(&sequence);
		createdCarSequences.pop_back();
	}
	for (int i = createdPeds.size(); i > 0; i--) {
		Entity e = createdPeds[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdPeds.pop_back();
	}
	for (int i = createdDrivers.size(); i > 0; i--) {
		Entity e = createdDrivers[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdDrivers.pop_back();
	}
	for (int i = createdVehicles.size(); i > 0; i--) {
		Entity e = createdVehicles[i - 1];
		ENTITY::DELETE_ENTITY(&e);
		createdVehicles.pop_back();
	}
}

void deleteCar(int i) {
	int numCars = createdVehicles.size();

	if (!createdCarSequences.empty()) {
		AI::CLEAR_SEQUENCE_TASK(&createdCarSequences[i]);
		std::swap(createdCarSequences[i], createdCarSequences[numCars - 1]);
		createdCarSequences.pop_back();
	}

	ENTITY::DELETE_ENTITY(&createdDrivers[i]);
	ENTITY::DELETE_ENTITY(&createdVehicles[i]);
	std::swap(createdDrivers[i], createdDrivers[numCars - 1]);
	std::swap(createdVehicles[i], createdVehicles[numCars - 1]);

	createdDrivers.pop_back();
	createdVehicles.pop_back();

}


void deletePed(int i) {
	int numPeds = createdPeds.size();

	if (!createdPedSequences.empty()) {
		AI::CLEAR_SEQUENCE_TASK(&createdPedSequences[i]);
		std::swap(createdPedSequences[i], createdPedSequences[numPeds - 1]);
		createdPedSequences.pop_back();
	}

	ENTITY::DELETE_ENTITY(&createdPeds[i]);
	std::swap(createdPeds[i], createdPeds[numPeds - 1]);
	createdPeds.pop_back();
}