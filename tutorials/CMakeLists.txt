cmake_minimum_required(VERSION 2.8.12)

project(dart-tutorials)

add_subdirectory(transferNet)
add_subdirectory(runSingleRect)
add_subdirectory(runRect3D)
add_subdirectory(runMultiLink)
add_subdirectory(genTrainDataRectC1)
add_subdirectory(genTrainDataRectC2)
add_subdirectory(genTrainDataRectR1)
add_subdirectory(genTrainDataRectR2)
add_subdirectory(genTrainDataPolyC1)
add_subdirectory(genTrainDataPolyC2)
add_subdirectory(genTrainDataPolyR1)
add_subdirectory(genTrainDataPolyR2)

add_subdirectory(trainRect3DC1);
add_subdirectory(trainRect3DC2);
add_subdirectory(trainRect3DC3);
add_subdirectory(trainRect3DR1);
add_subdirectory(trainRect3DR2);
add_subdirectory(trainRect3DR3);

add_subdirectory(customDataGen)
add_subdirectory(customRun)

add_subdirectory(PDDrunRect3D)
add_subdirectory(Metrics)
