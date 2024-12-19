#include "CommonTypes.h"

void SparseSurfelFusion::DatasetSwitch(DatasetType type, std::string& res, std::string& dis, std::string& speed, std::string& action)
{

    switch (type)
    {
    case DatasetType::LowRes_1000mm_calibration:
        res = "640_400";
        dis = "1.0m";
        speed = "";
        action = "calibration";
        break;
    case DatasetType::LowRes_1000mm_boxing_slow:
        res = "640_400";
        dis = "1.0m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::LowRes_1000mm_boxing_fast:
        res = "640_400";
        dis = "1.0m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::LowRes_1000mm_doll_slow:
        res = "640_400";
        dis = "1.0m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::LowRes_1000mm_doll_fast:
        res = "640_400";
        dis = "1.0m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::LowRes_1000mm_coat_slow:
        res = "640_400";
        dis = "1.0m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::LowRes_1000mm_coat_fast:
        res = "640_400";
        dis = "1.0m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::LowRes_1000mm_claphands_slow:
        res = "640_400";
        dis = "1.0m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::LowRes_1000mm_claphands_fast:
        res = "640_400";
        dis = "1.0m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::HighRes_1000mm_boxing_slow:
        res = "1280_720";
        dis = "1.0m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::HighRes_1000mm_boxing_fast:
        res = "1280_720";
        dis = "1.0m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::HighRes_1000mm_doll_slow:
        res = "1280_720";
        dis = "1.0m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::HighRes_1000mm_doll_fast:
        res = "1280_720";
        dis = "1.0m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::HighRes_1000mm_coat_slow:
        res = "1280_720";
        dis = "1.0m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::HighRes_1000mm_coat_fast:
        res = "1280_720";
        dis = "1.0m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::HighRes_1000mm_claphands_slow:
        res = "1280_720";
        dis = "1.0m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::HighRes_1000mm_claphands_fast:
        res = "1280_720";
        dis = "1.0m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::LowRes_1500mm_calibration:
        res = "640_400";
        dis = "1.5m";
        speed = "";
        action = "calibration";
        break;
    case DatasetType::LowRes_1500mm_boxing_slow:
        res = "640_400";
        dis = "1.5m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::LowRes_1500mm_boxing_fast:
        res = "640_400";
        dis = "1.5m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::LowRes_1500mm_doll_slow:
        res = "640_400";
        dis = "1.5m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::LowRes_1500mm_doll_fast:
        res = "640_400";
        dis = "1.5m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::LowRes_1500mm_coat_slow:
        res = "640_400";
        dis = "1.5m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::LowRes_1500mm_coat_fast:
        res = "640_400";
        dis = "1.5m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::LowRes_1500mm_claphands_slow:
        res = "640_400";
        dis = "1.5m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::LowRes_1500mm_claphands_fast:
        res = "640_400";
        dis = "1.5m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::LowRes_1500mm_ChallengeTest_A:
        res = "640_400";
        dis = "1.5m";
        speed = "fast";
        action = "ChallengeTest_A";
        break;
    case DatasetType::LowRes_1500mm_ChallengeTest_B:
        res = "640_400";
        dis = "1.5m";
        speed = "slow";
        action = "ChallengeTest_B";
        break;
    case DatasetType::HighRes_1500mm_boxing_slow:
        res = "1280_720";
        dis = "1.5m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::HighRes_1500mm_boxing_fast:
        res = "1280_720";
        dis = "1.5m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::HighRes_1500mm_doll_slow:
        res = "1280_720";
        dis = "1.5m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::HighRes_1500mm_doll_fast:
        res = "1280_720";
        dis = "1.5m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::HighRes_1500mm_coat_slow:
        res = "1280_720";
        dis = "1.5m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::HighRes_1500mm_coat_fast:
        res = "1280_720";
        dis = "1.5m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::HighRes_1500mm_claphands_slow:
        res = "1280_720";
        dis = "1.5m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::HighRes_1500mm_claphands_fast:
        res = "1280_720";
        dis = "1.5m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::LowRes_2000mm_calibration:
        res = "640_400";
        dis = "2.0m";
        speed = "";
        action = "calibration";
        break;
    case DatasetType::LowRes_2000mm_boxing_slow:
        res = "640_400";
        dis = "2.0m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::LowRes_2000mm_boxing_fast:
        res = "640_400";
        dis = "2.0m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::LowRes_2000mm_doll_slow:
        res = "640_400";
        dis = "2.0m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::LowRes_2000mm_doll_fast:
        res = "640_400";
        dis = "2.0m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::LowRes_2000mm_coat_slow:
        res = "640_400";
        dis = "2.0m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::LowRes_2000mm_coat_fast:
        res = "640_400";
        dis = "2.0m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::LowRes_2000mm_claphands_slow:
        res = "640_400";
        dis = "2.0m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::LowRes_2000mm_claphands_fast:
        res = "640_400";
        dis = "2.0m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::HighRes_2000mm_boxing_slow:
        res = "1280_720";
        dis = "2.0m";
        speed = "slow";
        action = "boxing";
        break;
    case DatasetType::HighRes_2000mm_boxing_fast:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "boxing";
        break;
    case DatasetType::HighRes_2000mm_doll_slow:
        res = "1280_720";
        dis = "2.0m";
        speed = "slow";
        action = "doll";
        break;
    case DatasetType::HighRes_2000mm_doll_fast:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "doll";
        break;
    case DatasetType::HighRes_2000mm_coat_slow:
        res = "1280_720";
        dis = "2.0m";
        speed = "slow";
        action = "coat";
        break;
    case DatasetType::HighRes_2000mm_coat_fast:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "coat";
        break;
    case DatasetType::HighRes_2000mm_claphands_slow:
        res = "1280_720";
        dis = "2.0m";
        speed = "slow";
        action = "claphands";
        break;
    case DatasetType::HighRes_2000mm_claphands_fast:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "claphands";
        break;
    case DatasetType::LowRes_2000mm_ChallengeTest_A:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "ChallengeTest_A";
        break;
    case DatasetType::LowRes_2000mm_ChallengeTest_B:
        res = "1280_720";
        dis = "2.0m";
        speed = "fast";
        action = "ChallengeTest_B";
        break;
    case DatasetType::LowRes_2000mm_ChallengeTest_C:
        res = "1280_720";
        dis = "2.0m";
        speed = "slow";
        action = "ChallengeTest_C";
        break;
    default:
        break;
    }
}
