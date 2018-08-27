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
        pos << 0,0,0, 0,h,0;
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
        vel[0] = 7.15459;
        vel[1] = 7.17186;
        vel[2] = 17.3877;

        vel[3] = -0.465992; //x vel
        vel[4] = 0.504826; //y vel
        vel[5] = 1.32386; //z vel
        
        
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
        
        //            std::cout << "vel_set:" << bNode->getSpatialVelocity(Frame::World(),Frame::World()).transpose() << std::endl;
        
        VelIn = bNode->getSpatialVelocity(Frame::World(),Frame::World());
        PosIn = bNode->getSkeleton()->getPositions();
        auto oldAng = bNode->getAngularMomentum();
        
        // check collision
        auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
        auto collisionGroup = mWorld->getConstraintSolver()->getCollisionGroup();
        dart::collision::CollisionOption option;
        dart::collision::CollisionResult result;
        bool collision = collisionGroup->collide(option, &result);

        if (collision && result.getNumContacts() == 4)
        {	
        	double delTime = mWorld->getTimeStep();
            // Standard LCP
            // mWorld->getConstraintSolver()->solve();

            // Replace it with our own testing solving
            auto pos1 = result.getContact(0).point;
            auto pos2 = result.getContact(1).point;
            auto pos3 = result.getContact(2).point;
            auto pos4 = result.getContact(3).point;
            // Impulse/time => Force
            double tempX = 2.53485;
            double tempZ = -181.461;
            double tempTY = -21.1335;

            Eigen::Vector3d friction;
            friction << tempX, 0.0, tempZ;
            Eigen::Vector3d torque;
            torque << 0.0, tempTY, 0.0;

	        bNode -> clearExternalForces();
	        mWorld->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
	        // middle of the point
	        bNode -> addExtForce(friction, (pos1+pos2+pos3+pos4)/4, false, false);
	        bNode -> addExtTorque(torque, false);
	        bNode->getSkeleton()->computeForwardDynamics();
	        bNode->getSkeleton()->integrateVelocities(mWorld->getTimeStep());
	        
	        bNode -> setFrictionCoeff(0.0);
	        mWorld->getSkeleton("ground skeleton")->getBodyNode(0) -> setFrictionCoeff(0.0);
	        mWorld->getConstraintSolver()->solve();

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
