#include "script.h"
#include "keyboard.h"
#include "menu.h"

#include <string>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <Windows.h>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

#pragma warning(disable : 4244 4305) // double <-> float conversions

json settings;
std::string settingFile;
std::vector<Vehicle> createdVehicles;
std::vector<Ped> createdPeds;
std::vector<int> createdPedSequences;
std::vector<Ped> createdDrivers;
std::vector<int> createdCarSequences;


void printOnScreen(std::string s) {
	DWORD maxTickCount = GetTickCount() + 5000;
	while (GetTickCount() < maxTickCount) {
		draw_menu_line(s, 350.0, 15.0, 15, 15, 5.0, false, true);
		WAIT(0);
	}
	WAIT(0);
}

std::vector<std::string> readAllFiles(std::string folder)
{
	std::vector<std::string> res;
	std::string path = folder + "\\*";
	WIN32_FIND_DATA data;
	HANDLE hFind = ::FindFirstFile(path.c_str(), &data);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			if (!(data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				res.push_back(data.cFileName);
			}
		} while (FindNextFile(hFind, &data));
		FindClose(hFind);
	}
	return res;
}

Entity spawnAtCoords(LPCSTR modelName, int modelType, Vector3 coords, float heading = 0.0)
{
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
	createdDrivers.push_back(p);

	if (modelName[0] != '\0') {
		STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(model);
	}

	return p;
}

void deleteAllCreated() {
	for (int i = createdPedSequences.size(); i > 0; i--) {
		int sequence = createdCarSequences.back();
		AI::CLEAR_SEQUENCE_TASK(&sequence);
		createdCarSequences.pop_back();
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
	AI::CLEAR_SEQUENCE_TASK(&createdCarSequences[i]);
	ENTITY::DELETE_ENTITY(&createdDrivers[i]);
	ENTITY::DELETE_ENTITY(&createdVehicles[i]);
	std::swap(createdDrivers[i], createdDrivers[numCars - 1]);
	std::swap(createdVehicles[i], createdVehicles[numCars - 1]);
	std::swap(createdCarSequences[i], createdCarSequences[numCars - 1]);

	createdCarSequences.pop_back();
	createdDrivers.pop_back();
	createdVehicles.pop_back();
}


void deletePed(int i) {
	int numPeds = createdPeds.size();
	AI::CLEAR_SEQUENCE_TASK(&createdPedSequences[i]);
	ENTITY::DELETE_ENTITY(&createdPeds[i]);
	std::swap(createdPeds[i], createdPeds[numPeds - 1]);
	std::swap(createdPedSequences[i], createdPedSequences[numPeds - 1]);

	createdPedSequences.pop_back();
	createdPeds.pop_back();
}


float randomNum(float low, float high) {
	return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

int randomNum(int low, int high) {
	return low + rand() % (high - low + 1);
}

float getModelLength(LPCSTR modelName) {
	Vector3 min, max;
	Hash model = GAMEPLAY::GET_HASH_KEY((char *)modelName);
	STREAMING::REQUEST_MODEL(model);
	while (!STREAMING::HAS_MODEL_LOADED(model)) WAIT(0);
	GAMEPLAY::GET_MODEL_DIMENSIONS(model, &min, &max);
	return max.y - min.y;
}

void clearArea() {
	GAMEPLAY::CLEAR_AREA_OF_PEDS(0, 0, 0, 10000, 1);
	GAMEPLAY::CLEAR_AREA_OF_VEHICLES(0, 0, 0, 10000, false, false, false, false, false);
}

void getCoords(Entity e, Vector3* coords3D = NULL, Vector2* coords2D = NULL, float* heading = NULL) {

	Vector3 coords = ENTITY::GET_ENTITY_COORDS(e, TRUE);

	if (coords3D != NULL) {
		*coords3D = coords;
	}

	if (coords2D != NULL) {
		GRAPHICS::_WORLD3D_TO_SCREEN2D(coords.x, coords.y, coords.z,
			&(coords2D->x), &(coords2D->y));
		int screen_w, screen_h;
		GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);
		coords2D->x *= screen_w;
		coords2D->y *= screen_h;
	}

	if (heading != NULL) {
		*heading = ENTITY::GET_ENTITY_HEADING(e);
	}

}

std::string showCoords() {
	Player e = PLAYER::PLAYER_PED_ID();
	Vector3 coords3D;
	Vector2 coords2D;
	float heading;

	getCoords(e, &coords3D, &coords2D, &heading);

	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << coords3D.x
		<< "," << coords3D.y << "," << coords3D.z << "," << coords2D.x << "," << coords2D.y << "," << heading;
	std::string s = stream.str();
	draw_menu_line(s, 350.0, 15.0, coords2D.y, coords2D.x, 5.0, false, true);
	return s;
}

std::string getKeyboardInput() {
	std::string res;
	// Invoke keyboard
	GAMEPLAY::DISPLAY_ONSCREEN_KEYBOARD(true, "", "", "", "", "", "", 140);
	// Wait for the user to edit
	while (GAMEPLAY::UPDATE_ONSCREEN_KEYBOARD() == 0) WAIT(0);
	// Make sure they didn't exit without confirming their change
	if (!GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT()) {
		return res;
	}
	else {
		return GAMEPLAY::GET_ONSCREEN_KEYBOARD_RESULT();
	}
}

void processSampling() {
	std::ofstream record("sampledCoords.csv");
	WAIT(200);
	Menu hintLine("Press 5 to sample a point. Press 0 to exit.", std::vector<std::string>{});
	while (true) {
		clearArea();
		hintLine.drawVertical(0);
		std::string coords = showCoords();
		if (IsKeyJustUp(VK_NUMPAD5)) {
			std::string label = getKeyboardInput();
			if (!label.empty())
				record << label << ","  << coords << std::endl;
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			record.close();
			WAIT(0);
			break;
		}
		WAIT(0);
	}
}

template<typename Terminator>
void processRecording(std::vector<std::string> fileName, Terminator terminator, int waitTime) {

	std::unordered_map<Entity, int> IDMap;

	std::string dirName;
	std::ofstream record;

	int screen_w, screen_h;
	GRAPHICS::GET_SCREEN_RESOLUTION(&screen_w, &screen_h);

	for (int i = 0; i < fileName.size() - 1; i++) {
		if (i != 0) {
			dirName += "/";
		}
		dirName += fileName[i];
		CreateDirectory(dirName.c_str(), NULL);
	}
	record.open(dirName + "/" + fileName.back());

	int count = 0;
	int carID = 0;
	int pedID = 100; // constant offset
	while (true) {
		bool seeACar = false;
		for (int i = 0; i < createdVehicles.size(); i++) {
			Entity e = createdVehicles[i];
			if (!IDMap.count(e)) {
				IDMap[e] = carID;
				carID++;
			}

			Vector3 curCoords = ENTITY::GET_ENTITY_COORDS(e, !ENTITY::IS_ENTITY_DEAD(e));
			Vector2 coords2D;
			GRAPHICS::_WORLD3D_TO_SCREEN2D(curCoords.x, curCoords.y, curCoords.z, &(coords2D.x), &(coords2D.y));

			if (coords2D.x < 0) {
				continue;
			}
			seeACar = true;

			coords2D.x *= screen_w;
			coords2D.y *= screen_h;

			record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << std::endl;
		}
		if (seeACar) {
			for (int i = 0; i < createdPeds.size(); i++) {
				Entity e = createdPeds[i];
				if (!IDMap.count(e)) {
					IDMap[e] = pedID;
					pedID++;
				}
				Vector3 curCoords = ENTITY::GET_ENTITY_COORDS(e, !ENTITY::IS_ENTITY_DEAD(e));
				Vector2 coords2D;
				GRAPHICS::_WORLD3D_TO_SCREEN2D(curCoords.x, curCoords.y, curCoords.z, &(coords2D.x), &(coords2D.y));

				if (coords2D.x < 0) {
					continue;
				}

				coords2D.x *= screen_w;
				coords2D.y *= screen_h;

				record << count << "," << IDMap[e] << "," << coords2D.x << "," << coords2D.y << std::endl;

			}
		}
		if (terminator()) { 
			break;
		}
		WAIT(waitTime);
		if (!seeACar && count == 0)
			continue;
		count++;
	}
}

void teleport() {

	if (settings.empty()) {
		return;
	}

	Ped p = PLAYER::PLAYER_PED_ID();
	ENTITY::SET_ENTITY_COORDS(p, settings["centerCoords"][0], settings["centerCoords"][1], settings["centerCoords"][2], 1, 0, 0, 1);
}

void setCars() {

	if (settings["totalCars"] == 0) return;

	std::string carModelStr = settings["carModel"].get<std::string>();
	std::vector<char> carModel(carModelStr.begin(), carModelStr.end());
	carModel.push_back('\0');

	float carLength = getModelLength(&carModel[0]);

	std::vector<int> numCars(settings["carStartPositions"].size(), 0);
	for (int i = 0; i < settings["totalCars"]; i++) {
		int lane = randomNum(0, numCars.size() - 1);
		numCars[lane]++;
	}

	for (int i = 0; i < settings["carStartPositions"].size(); i++) {
		Vector3 start;
		Vector3 front = Vector3(settings["carStartPositions"][i]["front"][0], settings["carStartPositions"][i]["front"][1], settings["carStartPositions"][i]["front"][2]);
		Vector3 back = Vector3(settings["carStartPositions"][i]["back"][0], settings["carStartPositions"][i]["back"][1], settings["carStartPositions"][i]["back"][2]);
		start.x = randomNum(front.x, back.x);
		float ratio = (start.x - front.x) / (back.x - front.x);
		start.y = (back.y - front.y) * ratio + front.y;
		start.z = (back.z - front.z) * ratio + front.z;

		ratio = (settings["carInterval"] + carLength) / SYSTEM::VDIST(front.x, front.y, front.z, back.x, back.y, back.z);

		for (int j = 0; j < numCars[i]; j++) {

			Vehicle v = spawnAtCoords(&carModel[0], MODEL_VEH, start, settings["carHeading"]);
			Ped driver = spawnDriver(v, settings["pedModel"].get<std::string>().c_str());

			int goal = randomNum(0, settings["carStartPositions"][i]["goals"].size() - 1);
			float speed = randomNum((float)settings["carMinSpeed"], settings["carMaxSpeed"]);

			int sequence = 0;
			AI::OPEN_SEQUENCE_TASK(&sequence);
			AI::TASK_VEHICLE_DRIVE_TO_COORD(0, v, settings["carStartPositions"][i]["goals"][goal][0], settings["carStartPositions"][i]["goals"][goal][1], settings["carStartPositions"][i]["goals"][goal][2], speed, 0, GAMEPLAY::GET_HASH_KEY(&carModel[0]), settings["driveStyle"], settings["carStopRange"], 1.0);
			AI::CLOSE_SEQUENCE_TASK(sequence);
			AI::TASK_PERFORM_SEQUENCE(driver, sequence);

			createdCarSequences.push_back(sequence);

			start.x = (back.x - front.x) * ratio + start.x;
			start.y = (back.y - front.y) * ratio + start.y;
			start.z = (back.z - front.z) * ratio + start.z;
			WAIT(0);
		}
	}
}

void setWanderPed() {
	std::vector<Vector3> pedsPosition;

	if (settings["totalWanderPeds"] == 0) return;

	Vector3 origin = Vector3(settings["wanderPedsGenArea"]["origin"][0], settings["wanderPedsGenArea"]["origin"][1], settings["wanderPedsGenArea"]["origin"][2]);
	Vector3 unit1 = Vector3(settings["wanderPedsGenArea"]["unit1"][0], settings["wanderPedsGenArea"]["unit1"][1], settings["wanderPedsGenArea"]["unit1"][2]);
	Vector3 unit2 = Vector3(settings["wanderPedsGenArea"]["unit2"][0], settings["wanderPedsGenArea"]["unit2"][1], settings["wanderPedsGenArea"]["unit2"][2]);

	float walkRadius = SYSTEM::VDIST(unit1.x, unit1.y, unit1.z, unit2.x, unit2.y, unit2.z) * 0.7;

	for (int i = 0; i < settings["totalWanderPeds"]; i++) {
		float ratio1 = randomNum(0.0f, 1.0f);
		float ratio2 = randomNum(0.0f, 1.0f);
		float x = ratio1 * (unit1.x - origin.x) + ratio2 * (unit2.x - origin.x) + origin.x;
		float y = ratio1 * (unit1.y - origin.y) + ratio2 * (unit2.y - origin.y) + origin.y;
		float z = ratio1 * (unit1.z - origin.z) + ratio2 * (unit2.z - origin.z) + origin.z;
		bool tooClose = false;
		for (int j = 0; j < pedsPosition.size(); j++) {
			if (SYSTEM::VDIST(pedsPosition[j].x, pedsPosition[j].y, pedsPosition[j].z, x, y, z) < 1.0) {
				tooClose = true;
				break;
			}
		}
		if (tooClose) {
			i--;
			continue;
		}
		pedsPosition.push_back(Vector3(x, y, z));
		Ped p = spawnAtCoords(settings["pedModel"].get<std::string>().c_str(), MODEL_PED, pedsPosition.back());


		int sequence = 0;
		AI::OPEN_SEQUENCE_TASK(&sequence);
		AI::TASK_WANDER_IN_AREA(0, settings["centerCoords"][0], settings["centerCoords"][1], settings["centerCoords"][2], walkRadius, 0.0, 0.0);
		AI::CLOSE_SEQUENCE_TASK(sequence);

		AI::TASK_PERFORM_SEQUENCE(p, sequence);
		createdPedSequences.push_back(sequence);

		//AI::TASK_WANDER_STANDARD(p, 10.0, 10);
		WAIT(0);
	}
}

void setScriptPed(bool onlyOnce = false) {
	std::vector<std::pair<Vector3, Vector3>> pedRoutes;

	int totalScriptPeds = onlyOnce ? 1 : settings["totalScriptPeds"];

	for (int i = 0; i < totalScriptPeds; i++) {
		int startArea = randomNum(0, settings["scriptPedsEndPoints"].size() - 1);
		int startCode = randomNum(0, settings["scriptPedsEndPoints"][startArea].size() - 1);
		Vector3 start = Vector3(settings["scriptPedsEndPoints"][startArea][startCode][0], settings["scriptPedsEndPoints"][startArea][startCode][1], settings["scriptPedsEndPoints"][startArea][startCode][2]);

		int endArea = randomNum(0, settings["scriptPedsEndPoints"].size() - 1);
		if (endArea == startArea) {
			i--;
			continue;
		}
		int endCode = randomNum(0, settings["scriptPedsEndPoints"][endArea].size() - 1);
		Vector3 end = Vector3(settings["scriptPedsEndPoints"][endArea][endCode][0], settings["scriptPedsEndPoints"][endArea][endCode][1], settings["scriptPedsEndPoints"][endArea][endCode][2]);

		bool repeated = false;
		for (int j = 0; j < pedRoutes.size(); j++) {
			if (start == pedRoutes[j].first || (end == pedRoutes[j].first && start == pedRoutes[j].second)) {
				repeated = true;
				break;
			}
		}

		if (repeated) {
			i--;
			continue;
		}

		pedRoutes.push_back(std::make_pair(start, end));
		Ped p = spawnAtCoords(settings["pedModel"].get<std::string>().c_str(), MODEL_PED, start);

		int sequence = 0;
		AI::OPEN_SEQUENCE_TASK(&sequence);
		AI::TASK_GO_STRAIGHT_TO_COORD(0, end.x, end.y, end.z, 1.0, 200000, 0.0, 0.2);
		AI::CLOSE_SEQUENCE_TASK(sequence);

		AI::TASK_PERFORM_SEQUENCE(p, sequence);
		createdPedSequences.push_back(sequence);

		WAIT(0);
	}
}

void startSimulation() {

	if (settings.empty()) {
		return;
	}

	DWORD nextTickCount = GetTickCount();
	
	auto terminator = [&]() {
		clearArea();

		if (settings["continuousPedGen"]) {
			if (GetTickCount() >= nextTickCount) {
				setScriptPed(true);
				nextTickCount = GetTickCount() + randomNum((int)settings["minPedGenInterval"], settings["maxPedGenInterval"]);
			}
		}

		for (int i = 0; i < createdVehicles.size(); i++) {
			if (AI::GET_SEQUENCE_PROGRESS(createdDrivers[i]) == -1) {
				deleteCar(i);
				i--;
			}
		}

		for (int i = 0; i < createdPeds.size(); i++) {
			if (AI::GET_SEQUENCE_PROGRESS(createdPeds[i]) == -1) {
				deletePed(i);
				i--;
			}
		}
		if (createdVehicles.size() == 0) {
			return true;
		}
		return false;
	};

	clearArea();
	setCars();
	setWanderPed();
	setScriptPed();
	processRecording(settings["recordDirectory"], terminator, settings["period"]);
	deleteAllCreated();
}

void toggleCamera() {

	if (settings.empty()) {
		return;
	}

	static bool initialized(false);
	static bool active(false);

	static Camera customCam = CAM::CREATE_CAM("DEFAULT_SCRIPTED_CAMERA", true);

	if (!initialized) {
		initialized = true;
		CAM::SET_CAM_COORD(customCam, settings["centerCoords"][0], settings["centerCoords"][1], (float)settings["centerCoords"][2] + settings["cameraHeight"]);
		CAM::SET_CAM_FOV(customCam, 130.0);
		// CAM::POINT_CAM_AT_COORD(customCam, centerCoords.x, centerCoords.y, centerCoords.z);
		CAM::SET_CAM_ROT(customCam, 270.0, 360.0 - (float)settings["carHeading"], 0.0, 2);
		CAM::SET_CINEMATIC_MODE_ACTIVE(false);
	}
	if (!active) {
		CAM::SET_CAM_ACTIVE(customCam, true);
		CAM::RENDER_SCRIPT_CAMS(true, false, 3000, true, false);
	}
	else {
		CAM::SET_CAM_ACTIVE(customCam, false);
		CAM::RENDER_SCRIPT_CAMS(false, false, 3000, true, false);
	}
	active = !active;
}

// TODO: enable multiple setting reading
// Need new menu
void loadSettings() {
	std::ifstream config("settings.json");
	config >> settings;
}

void recordPosition(json& saveSpot, bool appendBack = false, bool getHeading = false) {
	Menu hintLine("Press 5 to record coords. Press 0 to exit.", std::vector<std::string>{});
	while (true) {
		clearArea();
		hintLine.drawVertical(0);
		if (IsKeyJustUp(VK_NUMPAD5)) {
			if (!getHeading) {
				Vector3 centerCoords;
				getCoords(PLAYER::PLAYER_PED_ID(), &centerCoords);
				json coords_j = { centerCoords.x, centerCoords.y, centerCoords.z };
				if (appendBack == false) {
					saveSpot = coords_j;
				}
				else {
					saveSpot.push_back(coords_j);
				}
			}
			else {
				float heading;
				getCoords(PLAYER::PLAYER_PED_ID(), NULL, NULL, &heading);
				saveSpot = heading;
			}
		}
		if (IsKeyJustUp(VK_NUMPAD0)) {
			WAIT(0);
			break;
		}
		WAIT(0);
	}
}

Menu createSettingsMenu("Modify Settings", std::vector<std::string>
{"Save Settings",
"Center Coords",
"Add carStartPositions",
"Clear carStartPositions",
"car Heading",
"wanderPedsGenArea",
"Add scriptPedsEndPoints",
"Clear scriptPedsEndPoints"
});

// TODO: Guide through creation
void modifySettings() {

	json settingHolder;

	if (settings.empty()) {
		settingHolder["centerCoords"] = { -1382.03, -390.19, 36.69 };
		settingHolder["carStartPositions"] = { { { "front", {-1420.16,-421.93,36.22} }, { "back", {-1427.47,-426.17,36.05} }, { "goals", { { -1429.17,-327.34,44.43 }, { -1320.46,-361.92,36.75 } } } }, { { "front", {-1417.19,-426.05,36.14} }, { "back", {-1425.10,-430.49,36.00} }, { "goals", { { -1320.16,-489.37,33.34 }, { -1319.56, -367.42, 36.69 } } } } };
		settingHolder["wanderPedsGenArea"] = { { "origin", {-1362.92, -394.49, 36.55} }, { "unit1", {-1387.76, -411.07, 36.56} }, { "unit2", {-1378.92, -370.83, 36.51} } };
		settingHolder["scriptPedsEndPoints"] = { { {-1377.41,-366.81,36.60}, {-1380.84,-366.67,36.57} }, { {-1409.80,-386.40,36.63}, {-1408.43,-385.78,36.61} }, { {-1390.06,-415.37,36.58}, {-1385.91,-417.79,36.61} }, { {-1359.15,-397.74,36.61}, {-1357.56,-393.77,36.58} } };
		settingHolder["recordDirectory"] = json::array({ "complexScene", "record.csv" });
		settingHolder["cameraHeight"] = 15.0;
		settingHolder["period"] = 100;
		settingHolder["totalCars"] = 3;
		settingHolder["carHeading"] = 302.37;
		settingHolder["carInterval"] = 2.0;
		settingHolder["carStopRange"] = 2.0;
		settingHolder["driveStyle"] = 786475; // normal except 128--stop on lights
		settingHolder["carMinSpeed"] = 3.0;
		settingHolder["carMaxSpeed"] = 10.0;
		settingHolder["totalWanderPeds"] = 6;
		settingHolder["totalScriptPeds"] = 3;
		settingHolder["carModel"] = "Adder";
		settingHolder["pedModel"] = "";
		settingHolder["continuousPedGen"] = true;
		settingHolder["minPedGenInterval"] = 2000;
		settingHolder["maxPedGenInterval"] = 6000;
	}
	else {
		settingHolder = settings;
	}
	int lineActive = 0;

	DWORD waitTime = 150;

	while (true) {
		DWORD maxTickCount = GetTickCount() + waitTime;
		do
		{
			createSettingsMenu.drawVertical(lineActive);
			WAIT(0);
		} while (GetTickCount() < maxTickCount);

		waitTime = 0;

		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			switch (lineActive)
			{
			case 0: {
				std::ofstream config("settings.json");
				config << settingHolder;
				config.close();
				settings = settingHolder;
				break;
			}
			case 1: {
				recordPosition(settingHolder["centerCoords"]);
				break;
			}
			case 2: {
				json newPosition;
				printOnScreen("Record front of lane");
				recordPosition(newPosition["front"]);
				printOnScreen("Record back of lane");
				recordPosition(newPosition["back"]);
				printOnScreen("Record goals of lane.");
				recordPosition(newPosition["goals"], true);
				settingHolder["carStartPositions"].push_back(newPosition);
				break;
			}
			case 3: {
				settingHolder["carStartPositions"] = json();
				break;
			}
			case 4: {
				recordPosition(settingHolder["carHeading"], false, true);
				break;
			}
			case 5: {
				printOnScreen("Record center corner of area.");
				recordPosition(settingHolder["wanderPedsGenArea"]["origin"]);
				printOnScreen("Record side corner 1 of area.");
				recordPosition(settingHolder["wanderPedsGenArea"]["unit1"]);
				printOnScreen("Record side corner 2 of area.");
				recordPosition(settingHolder["wanderPedsGenArea"]["unit2"]);
				break;
			}
			case 6: {
				json oneSet;
				printOnScreen("Record points on one corner of sidewalk.");
				recordPosition(oneSet, true);
				if (!oneSet.empty()) {
					settingHolder["scriptPedsEndPoints"].push_back(oneSet);
				}
				break;
			}
			case 7: {
				settingHolder["scriptPedsEndPoints"] = json();
			}
			}
			waitTime = 200;
		}
		else if (bBack) {
			break;
		}
		else if (bUp) {
			if (lineActive == 0)
				lineActive = createSettingsMenu.lineCount();
			lineActive--;
			waitTime = 150;
		}
		else if (bDown)
		{
			lineActive++;
			if (lineActive == createSettingsMenu.lineCount())
				lineActive = 0;
			waitTime = 150;
		}
	}
}

Menu mainMenu("Main Menu", std::vector<std::string>
{"Sample Points",
"Teleport to Scene",
"Toggle Camera",
"Start Simulation",
"Modify Settings",
"Load Settings"});

void processMainMenu() {
	int lineActive = 0;

	DWORD waitTime = 150;

	while (true) {
		DWORD maxTickCount = GetTickCount() + waitTime;
		do
		{
			mainMenu.drawVertical(lineActive);
			WAIT(0);
		} while (GetTickCount() < maxTickCount);

		waitTime = 0;

		bool bSelect, bBack, bUp, bDown;
		get_button_state(&bSelect, &bBack, &bUp, &bDown, NULL, NULL);
		if (bSelect)
		{
			switch (lineActive)
			{
			case 0:
				processSampling();
				break;
			case 1:
				teleport();
				break;
			case 2:
				toggleCamera();
				break;
			case 3:
				startSimulation();
				break;
			case 4:
				modifySettings();
				break;
			case 5:
				loadSettings();
				break;
			}
			waitTime = 200;
		}
		else if (bBack) {
			break;
		}
		else if (bUp) {
			if (lineActive == 0)
				lineActive = mainMenu.lineCount();
			lineActive--;
			waitTime = 150;
		}
		else if (bDown)
		{
			lineActive++;
			if (lineActive == mainMenu.lineCount())
				lineActive = 0;
			waitTime = 150;
		}
	}
}


void main() {
	while (true) {
		if (IsKeyJustUp(VK_F6)) {
			AUDIO::PLAY_SOUND_FRONTEND(-1, "NAV_UP_DOWN", "HUD_FRONTEND_DEFAULT_SOUNDSET", 0);
			processMainMenu();
		}
		WAIT(0);
	}
}

void ScriptMain() {
	srand(GetTickCount());
	main();
}
