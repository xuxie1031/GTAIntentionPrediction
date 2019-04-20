#pragma once

#include "script.h"


namespace Sampling {
	float sampleRotation(std::string helpText, Entity referenceEntity);
	std::vector<Vector3d> samplePoints(int number, std::string helpText, const Color markerColor, const std::vector<Marker>& referencePoints = {}, const std::vector<Vector3d>& originalSet = {});
	CameraParams sampleCameraParams(std::string helpText, const std::vector<Marker>& referencePoints = {}, CameraParams original = CameraParams());
}