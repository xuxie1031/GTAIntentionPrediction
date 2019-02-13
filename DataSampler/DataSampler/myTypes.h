#pragma once
#include "../../inc/types.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

struct Vector3d : Vector3
{
	inline Vector3d& operator=(const Vector3d& other) // copy assignment
	{
		if (this != &other) { // self-assignment check expected
			this->x = other.x;
			this->y = other.y;
			this->z = other.z;
		}
		return *this;
	}

	inline Vector3d& operator=(const Vector3& other) // copy assignment
	{
		if (this != &other) { // self-assignment check expected
			this->x = other.x;
			this->y = other.y;
			this->z = other.z;
		}
		return *this;
	}

	inline bool operator==(const Vector3d& other) {
		return x == other.x && y == other.y && z == other.z;
	}

	inline Vector3d() {
		x = 0;
		y = 0;
		z = 0;
	}
	inline Vector3d(const float X,const float Y,const float Z) {
		x = X;
		y = Y;
		z = Z;
	}

	inline Vector3d(const Vector3& other) {
		operator=(other);
	}

	inline Vector3d(const json& vectorArray) {
		if (!vectorArray.is_array() || vectorArray.size() != 3) {
			throw std::invalid_argument("Invalid setting format for coords.");
		}
		x = vectorArray[0];
		y = vectorArray[1];
		z = vectorArray[2];
	}

	inline Vector3d operator+(const Vector3d& A) const
	{
		return Vector3d(x + A.x, y + A.y, z + A.z);
	}

	inline Vector3d operator-(const Vector3d& A) const
	{
		return Vector3d(x - A.x, y - A.y, z - A.z);
	}

	inline Vector3d operator/(const float A)
	{
		return Vector3d(x / A, y / A, z / A);
	}

	inline Vector3d operator*(const float A)
	{
		return Vector3d(x * A, y * A, z * A);
	}
};




struct Vector2
{
	float x;
	float y;
};