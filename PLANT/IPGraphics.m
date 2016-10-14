% InvertedPendulum Graphics
classdef IPGraphics < handle
    properties (GetAccess=private)
        fig
        guiAxes
        
        poleLength
        cartWidth
        cartHeight
        
        LEFT_LINE
        RIGHT_LINE
        TOP_LINE
        BOT_LINE
        POLE_LINE
        ANGLE_DISPLAY
    end
    
    methods
        
        %Constructor
        function obj = IPGraphics(theta) 
            obj.fig = figure();
            obj.guiAxes = axes;
            
            obj.poleLength = 1;
            
            set(obj.guiAxes, 'Visible', 'off');
            %# Stretch the axes over the whole figure.
            set(obj.guiAxes, 'Position', [0, 0, 1, 1]);
            %# Switch off autoscaling.
            set(obj.guiAxes, 'Xlim', [-obj.poleLength*1.2, obj.poleLength*1.2], 'YLim', [-obj.poleLength*1.2, obj.poleLength*1.2]);

            thetaUp = theta + deg2rad(90);
            obj.LEFT_LINE    = line([-.05, -.05], [-.05,  .05], 'Parent', obj.guiAxes);
            obj.RIGHT_LINE   = line([ .05,  .05], [-.05,  .05], 'Parent', obj.guiAxes);
            obj.TOP_LINE     = line([-.05,  .05], [ .05,  .05], 'Parent', obj.guiAxes);
            obj.BOT_LINE     = line([-.05,  .05], [-.05, -.05], 'Parent', obj.guiAxes);

            obj.POLE_LINE    = line([0, obj.poleLength*cos(thetaUp)], [0, obj.poleLength*sin(thetaUp)], 'Parent', obj.guiAxes);
        end
        
        %% update
        % theta - angle in degrees (where straight up is 90 degrees)
        % x     - location between of the center of the cart
        function update(obj, theta)
            
            delete(obj.POLE_LINE);
            delete(obj.ANGLE_DISPLAY);
            
            thetaUp = theta + (90 * pi/180);
            
            obj.POLE_LINE    = line([0, 0 + obj.poleLength*cos(thetaUp)], [0, (0 + obj.poleLength)*sin(thetaUp)], 'Parent', obj.guiAxes);

            obj.ANGLE_DISPLAY = uicontrol('Parent', obj.fig, 'Style', 'edit', 'String', -1*theta*180/pi);
%             uicontrol('Parent', obj.fig, 'Style', 'edit', 'String', x, 'Position', [0.9 0.2 0.1 0.1]);
            
            pause(0.0001)
        end
        
    end
end
        
