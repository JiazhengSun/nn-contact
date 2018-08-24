/*
 * Copyright (c) 2015-2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2015-2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <iostream>
#include <fstream>
//#include "mat.h"
//
//MATFile *pmat;
//mxArray *pa;
//mxArray *pb;

#define NUMSAM 50000
#define INWIDTH 5
#define OUTWIDTH 1 // r1 has only 2 DoF, the location of contact force is fixed
// for now we only regress the friction part
#define UNSYMCONE false

//const char *file = "training_r1_unsym_v1.mat";

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

const char* path;
ofstream file;

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);

        bNode = mWorld->getSkeleton("hopper")->getBodyNode(0);
        
        sampleCount = 0;
        ts = 0;
    }
    
    void keyboard(unsigned char key, int x, int y) override
    {
        switch(key)
        {
            default:
                SimWindow::keyboard(key, x, y);
        }
    }
    
    void drawWorld() const override
    {
        // Make sure lighting is turned on and that polygons get filled in
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        SimWindow::drawWorld();
    }
    
    void timeStepping() override
    {
        Eigen::Vector6d pos;
        auto Pi = dart::math::constants<double>::pi();
        double theta = dart::math::random(-Pi,Pi);
//        double theta = Pi/6.0;
        auto h = 0.1 * std::abs(sin(theta)) + 0.05*std::abs(cos(theta));
        pos << 0, 0, theta, 0, h-(1e-7), 0;
        bNode->getSkeleton()->getJoint(0)->setPositions(pos);
        
        
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[2] = dart::math::random(-50,50);
        vel[3] = dart::math::random(-10,10);
        vel[4] = dart::math::random(-7,3);
//        vel[2] = -30;
//        vel[3] = 4;
//        vel[4] = -2;
        
        // Create reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = bNode->getSkeleton()->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], 0.0);
        Eigen::Vector3d w = vel[2] * Eigen::Vector3d::UnitZ();
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // ?
        ref.setRelativeTransform(bNode->getTransform(&center));
        bNode->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        
//            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;

        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
        PosIn = bNode->getSkeleton()->getPositions();
        
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        
        if (collision && result.getNumContacts() == 2)
        {
            if (UNSYMCONE)
            {
                auto p1 = result.getContact(0).point;
                auto localp1 =  bNode->getWorldTransform().inverse() * p1;
                Eigen::Vector3d Vcp1;
                Vcp1 = bNode->getLinearVelocity(Eigen::Vector3d(localp1[0], localp1[1], 0.0),Frame::World());
                if (Vcp1[0] > 0)
                {
                    bNode->setFrictionCoeff(0.75);
                    mWorld->getSkeleton("ground skeleton")->getBodyNode(0)->setFrictionCoeff(0.75);
                }
                else
                {
                    bNode->setFrictionCoeff(1.5);
                    mWorld->getSkeleton("ground skeleton")->getBodyNode(0)->setFrictionCoeff(1.5);
                }
            }
            
//            SimWindow::timeStepping();
            
            // Integrate velocity for unconstrained skeletons
//            for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
//            {
//                auto skel = mWorld->getSkeleton(i);
//                if (!skel->isMobile())
//                    continue;
//                std::cout << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
//                skel->computeForwardDynamics();
//                std::cout << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
//                skel->integrateVelocities(mWorld->getTimeStep());
//                std::cout << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
//            }
            
            setbuf(stderr, buf);
            mWorld->getConstraintSolver()->solve();
            if (strlen(buf) > 0)
                {std::cerr << "omit!" << std::endl;}
            else
            {

                // Compute velocity changes given constraint impulses
                for (size_t i=0; i < mWorld->getNumSkeletons(); i++)
                {
                    auto skel = mWorld->getSkeleton(i);
                    if (!skel->isMobile())
                        continue;
                    
                    if (skel->isImpulseApplied())
                    {
                        skel->computeImpulseForwardDynamics();
                        skel->setImpulseApplied(false);
                    }
                }
                
                auto force1 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(0).force;

                if (force1.norm() < 1e-3)
                {
    //                storeOneInFile(2.0);
                }
                else // non-trivial contact impulse
                {
                    auto initialPoint1 = result.getContact(0).point;
                    auto localPoint1 =  bNode->getWorldTransform().inverse() * initialPoint1;
                    
                    Eigen::Vector3d Vcp;
                    Vcp = bNode->getLinearVelocity(Eigen::Vector3d(localPoint1[0], localPoint1[1], 0.0),Frame::World(),Frame::World());
    //                std::cout << Vcp.transpose() << std::endl;
                    
                    if (Vcp.norm() < 5e-4)
                    {
    //                    storeOneInFile(1.0);
                    }
                    else
                    {
    //                    std::cout << bNode->getConstraintImpulse().transpose() << std::endl;
                        Eigen::Vector3d newVel = bNode->getCOMLinearVelocity(Frame::World(),Frame::World());
                        auto fric = (newVel[0] - VelIn[3])/mWorld->getTimeStep();
                        storeOneForceInFile(fric);
                    }
                    
                }
            }
            
            buf[0] = '\0';
        }
        else
        {
            std::cout << "invalid: continue" << std::endl;
            SimWindow::timeStepping();
        }
        
        ts++;
    }
    
    void storeOneForceInFile(double friction)
    {
        if (sampleCount < NUMSAM)
        {
            Eigen::Vector6d InOut = Eigen::Vector6d::Zero();
            InOut << sin(PosIn[2]), cos(PosIn[2]), VelIn[2]/10.0, VelIn[3], VelIn[4], friction;
            std::cout << InOut.transpose() << std::endl;
            
            for (int i = 0; i < 6; i++) {
                file<< InOut[i];
                file<<",";
            }
            file<<"\n";
//            std::cout << "r "<< friction <<std::endl;
//            double* velarray = VelIn.data();
//            //scale down theta_dot by 10
//            velarray[2] = velarray[2]/10.0;
//            //            double* posarray = PosIn.data();
//            double posSin[2] = {sin(PosIn[2]), cos(PosIn[2])};
//            memcpy((void *)(mxGetPr(pa) + sampleCount*INWIDTH), (void *)(posSin), sizeof(double)*2);
//            memcpy((void *)(mxGetPr(pa) + sampleCount*INWIDTH+2), (void *)(velarray+2), sizeof(double)*3);
//            
//            memcpy((void *)(mxGetPr(pb) + sampleCount*OUTWIDTH), (void *)(&friction), sizeof(double));
            
            sampleCount++;
        }
        if (sampleCount == NUMSAM)
        {
            file.close();
//            int status = matPutVariable(pmat, "In", pa);
//            if (status != 0) {
//                printf("Error using matPutVariable a\n");
//                exit(1);
//            }
//            
//            mxDestroyArray(pa);
//            
//            status = matPutVariable(pmat, "Out", pb);
//            if (status != 0) {
//                printf("Error using matPutVariable b\n");
//                exit(1);
//            }
//            
//            mxDestroyArray(pb);
//            
//            if (matClose(pmat) != 0) {
//                printf("Error closing file %s\n",file);
//                exit(1);
//            }
            printf("Done\n");
            
            sampleCount++;
        }
    }
    
    BodyNodePtr bNode;
    Marker* m1;
    
    int ts;
    int sampleCount;
    
    Eigen::Vector6d VelIn;
    Eigen::Vector6d PosIn;
    
    char buf[BUFSIZ];
    
//    Eigen::Vector3d initialPoint1;
//    Eigen::Vector3d initialPoint2;
    
protected:
};



int main(int argc, char* argv[])
{
    if (UNSYMCONE) {
        path = "/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/train_data/rect_r1_unsym.csv";   
    } else {
        path = "/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/train_data/rect_r1_sym.csv";
    }
    file.open(path);

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
//    printf("Creating file %s...\n\n", file);
//    pmat = matOpen(file, "w");
//    if (pmat == NULL) {
//        printf("Error creating file %s\n", file);
//        exit(1);
//    }
//    
//    pa = mxCreateDoubleMatrix(INWIDTH,NUMSAM,mxREAL);
//    if (pa == NULL) {
//        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
//        printf("Unable to create mxArray.\n");
//        exit(1);
//    }
//    
//    
//    pb = mxCreateDoubleMatrix(OUTWIDTH,NUMSAM,mxREAL);
//    if (pb == NULL) {
//        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
//        printf("Unable to create mxArray.\n");
//        exit(1);
//    }
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
