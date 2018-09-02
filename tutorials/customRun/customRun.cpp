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
        pos <<0,0,0,0,0.0499999,0;
        mWorld->getSkeleton("hopper")->getJoint(0)->setPositions(pos);
        mWorld->getSkeleton("hopper")->getJoint(0)->setVelocities(Eigen::Vector6d::Zero());
        
        Eigen::Vector6d vel = Eigen::Vector6d::Zero();
        vel[0] = -17.8615; //wx
        vel[1] = 1.18801; //wy
        vel[2] = 6.84598; //wz

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
        
        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
        PosIn = bNode->getSkeleton()->getPositions();
        
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);

        if (collision && result.getNumContacts() == 4)
        {
            cout<<"pre-contact velocity is: "<<bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose()<<endl;
            // Replace it with our own testing solving
            
            double fx = 1515.57;
            double fz = 1515.57;
            double ty = -22.9489;
            
            Eigen::Vector3d friction;
            friction << fx, 0.0, fz;
            cout<<"Posiiton is: "<<PosIn.transpose()<<endl;
            Eigen::Vector3d torque;
            torque <<  -PosIn[4]*fz, ty, PosIn[4]*fx;
            
            bNode -> clearExternalForces();
            bNode -> clearConstraintImpulse();
            
            mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
            bNode -> addExtForce(friction, Eigen::Vector3d::Zero(), false, true);
            bNode -> addExtTorque(torque, false);
            
            bNode -> addConstraintImpulse(bNode->getAspectState().mFext * mWorld->getTimeStep());
            bNode -> clearExternalForces();
            bNode->getSkeleton()->computeImpulseForwardDynamics();
            
            bNode -> setFrictionCoeff(0.0);
            mWorld->getSkeleton("ground skeleton")->getBodyNode(0) -> setFrictionCoeff(0.0);
            mWorld->getConstraintSolver()->solve();
            //
            // Forward dynamics after solving LCP
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
            
	        Eigen::Vector6d new_vel = bNode->getSpatialVelocity(Frame::World(),Frame::World());
            cout<<"New velocities are: "<<new_vel.transpose()<<endl;
        }
        else
        {
            SimWindow::timeStepping();
        }
        
        ts++;
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
