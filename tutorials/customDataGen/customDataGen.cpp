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

//#define NUMSAM 500000
#define NUMSAM 1
#define UNSYMCONE false

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

        double h = 1.0;
        pos << 0,2.40961,0, 0,h,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(Eigen::Vector6d::Zero());
        
        Eigen::MatrixXd vts(3,8);
        vts << 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1,
               0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05, 
               0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1;

        double miny = 999;
        for (int i = 0; i < 8; i++) {
            Eigen::Vector3d vt = vts.col(i);
            Eigen::Vector3d offset(0,0,0);
            auto vt_pos = bNode->getWorldTransform()*(vt + offset);
            if (vt_pos[1]<miny) {
                miny = vt_pos[1];
            }
        }

        pos << 0,0,0, 0,h-(1e-7)-miny,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[0] = -0.893077*20.0; //wx
        vel[1] = 0.0594004*20.0; //wy
        vel[2] = 0.342299*20.0; //wz
        
        vel[3] = -4.92302; //x vel
        vel[4] = -3.03119; //y vel
        vel[5] = -4.33158; //z vel
        
        // Create reference frames for setting the initial velocity
        Eigen::Isometry3d centerTf(Eigen::Isometry3d::Identity());
        centerTf.translation() = bNode->getSkeleton()->getCOM();
        SimpleFrame center(Frame::World(), "center", centerTf);
        Eigen::Vector3d v = Eigen::Vector3d(vel[3], vel[4], vel[5]);
        Eigen::Vector3d w = Eigen::Vector3d(vel[0], vel[1], vel[2]);
        center.setClassicDerivatives(v, w);
        SimpleFrame ref(&center, "root_reference");
        // ?
        ref.setRelativeTransform(bNode->getTransform(&center));
        bNode->getSkeleton()->getJoint(0)->setVelocities(ref.getSpatialVelocity());
        Eigen::Vector3d oldAng = bNode->getWorldTransform().linear() * bNode->getAngularMomentum();
        
        //            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
        PosIn = bNode->getSkeleton()->getPositions();
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);
        if (collision && result.getNumContacts() == 4)
        {

            setbuf(stderr, buf);
            cout<<"pre-contact velocity is: "<<bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose()<<endl;
            VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
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
                auto force2 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(1).force;
                auto force3 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(2).force;
                auto force4 = mWorld->getConstraintSolver()->getLastCollisionResult().getContact(3).force;
                
                if (force1.norm() < 1e-3 && force2.norm() < 1e-3 && force3.norm() < 1e-3 && force4.norm() < 1e-3)
                {
                    cout<<"DETACHED"<<endl;
                }
                else // non-trivial contact impulse
                {
                    auto initialPoint1 = result.getContact(0).point;
                    auto localPoint1 =  bNode->getWorldTransform().inverse() * initialPoint1;
                    
                    auto initialPoint2 = result.getContact(1).point;
                    auto localPoint2 =  bNode->getWorldTransform().inverse() * initialPoint2;

                    auto initialPoint3 = result.getContact(2).point;
                    auto localPoint3 =  bNode->getWorldTransform().inverse() * initialPoint3;
                    
                    auto initialPoint4 = result.getContact(3).point;
                    auto localPoint4 =  bNode->getWorldTransform().inverse() * initialPoint4;
                    
                    Eigen::Vector3d Vcp1, Vcp2, Vcp3, Vcp4;
                    
                    Vcp1 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint1[0], localPoint1[1], localPoint1[2]),Frame::World(),Frame::World());
                    Vcp2 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint2[0], localPoint2[1], localPoint2[2]),Frame::World(),Frame::World());
                    Vcp3 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint3[0], localPoint3[1], localPoint3[2]),Frame::World(),Frame::World());
                    Vcp4 = bNode->getLinearVelocity(Eigen::Vector3d(localPoint4[0], localPoint4[1], localPoint4[2]),Frame::World(),Frame::World());
                    
                    if (Vcp1.norm() < 5e-4 && Vcp2.norm() < 5e-4 && Vcp3.norm() < 5e-4 && Vcp4.norm() < 5e-4) // both point's velocity all small, static
                    {
                        cout<<"Static!!!!"<<endl;
                    }
                    else
                    {   cout<<"DYNAMICCCC!"<<endl;
                        NewVel = bNode->getSpatialVelocity(Frame::World(),Frame::World());
                        Eigen::Vector3d newAng = bNode->getWorldTransform().linear() * bNode->getAngularMomentum();
                        auto fx = (NewVel[3] - VelIn[3])/mWorld->getTimeStep(); // linear impulse in one time step
                        auto fz = (NewVel[5] - VelIn[5])/mWorld->getTimeStep();
                        double t_y = (newAng[1] - oldAng[1])/mWorld->getTimeStep(); // Impulse = delMomentum. Momentum = I*w,
                        storeOneForceInFile(fx, fz, t_y);
                    }
                }
            }
            
            buf[0] = '\0';
        }
        else
        {
            SimWindow::timeStepping();
        }
        
        ts++;
    }
    
    void storeOneForceInFile(double fx, double fz, double t_y)
    {
        if (sampleCount < NUMSAM)
        {
            Eigen::Vector3d InOut_1 = Eigen::Vector3d::Zero();
            Eigen::Vector6d InOut_2 = Eigen::Vector6d::Zero();
            Eigen::Vector3d Result = Eigen::Vector3d::Zero();
            InOut_1 << PosIn[0], PosIn[1], PosIn[2];
            InOut_2 <<
                    VelIn[0], VelIn[1],VelIn[2], //angular vels
                    VelIn[3], VelIn[4], VelIn[5]; //linear vels
            Result << fx, fz, t_y;
            cout<<"Contact position is: "<<PosIn.transpose()<<endl;
            cout<<"New velocity is: "<<NewVel.transpose()<<endl;
            cout<<"Result Forces are: "<<Result.transpose()<<endl;
            sampleCount++;
        }
        if (sampleCount >= NUMSAM)
        {
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
    Eigen::Vector6d NewVel;
    
    char buf[BUFSIZ];
    
protected:
};

int main(int argc, char* argv[])
{

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    
    MyWindow window(world);
    
    std::cout << "space bar: simulation on/off" << std::endl;
    
    glutInit(&argc, argv);
    window.initWindow(640, 480, "Simple Test");
    glutMainLoop();
}
