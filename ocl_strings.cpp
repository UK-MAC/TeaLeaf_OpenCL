#include "ocl_strings.hpp"

#include <algorithm>
#include <sstream>
#include <iostream>

std::string matchParam
(std::ifstream& input, const char* param_name)
{
    std::string param_string("NO_SETTING");
    std::string line;
    /* read in line from file */
    while (getline(input, line))
    {
        if (line.find("!") != std::string::npos) continue;
        /* if it has the parameter name, its the line we want */
        if (line.find(param_name) != std::string::npos)
        {
            if (line.find("=") != std::string::npos)
            {
                param_string = std::string(line.erase(0, line.find("=")));
            }
            else
            {
                param_string = std::string(param_name);
            }
            break;
        }
    }

    return param_string;
}

std::string settingRead
(std::ifstream& input, const char * setting)
{
    std::string plat_name = matchParam(input, setting);

    // convert to lower case
    std::transform(plat_name.begin(),
                   plat_name.end(),
                   plat_name.begin(),
                   tolower);

    return plat_name;
}

int typeMatch
(std::string& type_name)
{
    //fprintf(stderr, "Matching with %s\n", type_name.c_str());

    // match
    if (type_name.find("cpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_CPU;
    }
    else if (type_name.find("gpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_GPU;
    }
    else if (type_name.find("accelerator") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ACCELERATOR;
    }
    else if (type_name.find("all") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else if (type_name.find("no_setting") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else
    {
        return 0;
    }
}

std::string strType
(cl_device_type dtype)
{
    switch (dtype)
    {
    case CL_DEVICE_TYPE_GPU :
        return std::string("GPU");
    case CL_DEVICE_TYPE_CPU :
        return std::string("CPU");
    case CL_DEVICE_TYPE_ACCELERATOR :
        return std::string("ACCELERATOR");
    default :
        return std::string("Device type does not match known values");
    }
}

bool paramEnabled
(std::ifstream& input, const char* param)
{
    std::string param_string = matchParam(input, param);
    return (param_string.find("NO_SETTING") == std::string::npos);
}

int preferredDevice
(std::ifstream& input)
{
    std::string param_string = matchParam(input, "opencl_device");

    int preferred_device;

    if (param_string.size() == 0)
    {
        // not found in file
        preferred_device = -1;
    }
    else
    {
        std::stringstream converter(param_string);

        if (!(converter >> preferred_device))
        {
            preferred_device = -1;
        }
    }

    return preferred_device;
}

