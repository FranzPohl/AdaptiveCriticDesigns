% Cart Pole Graphics
classdef CPGraphics < handle
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
        function obj = CPGraphics(x, theta) 
            obj.fig = figure;
            obj.guiAxes = axes;
            
            obj.poleLength = 1;           
            obj.cartWidth  = 0.5;
            obj.cartHeight = 0.2;
            
            set(obj.guiAxes, 'Visible', 'off');
            %# Stretch the axes over the whole figure.
            set(obj.guiAxes, 'Position', [0, 0, 1, 1]);
            %# Switch off autoscaling.
            set(obj.guiAxes, 'Xlim', [-obj.poleLength*3, obj.poleLength*3], 'YLim', [-obj.poleLength*3, obj.poleLength*3]);
          
            thetaUp = theta + deg2rad(90);
            
            obj.LEFT_LINE    = line([x - obj.cartWidth/2, x - obj.cartWidth/2], [0, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.RIGHT_LINE   = line([x + obj.cartWidth/2, x + obj.cartWidth/2], [0, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.TOP_LINE     = line([x - obj.cartWidth/2, x + obj.cartWidth/2], [obj.cartHeight, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.BOT_LINE     = line([x - obj.cartWidth/2, x + obj.cartWidth/2], [0, 0], 'Parent', obj.guiAxes);
            obj.POLE_LINE    = line([x, x + obj.poleLength*cos(thetaUp)], [obj.cartHeight, (obj.cartHeight + obj.poleLength)*sin(thetaUp)], 'Parent', obj.guiAxes);
        end
        
        %% update
        % theta - angle in degrees (where straight up is 90 degrees)
        % x     - location between of the center of the cart
        function update(obj,  x, theta)
            
            delete(obj.POLE_LINE);
            delete(obj.LEFT_LINE);
            delete(obj.RIGHT_LINE);
            delete(obj.TOP_LINE);
            delete(obj.BOT_LINE);
            delete(obj.ANGLE_DISPLAY);
            
            thetaUp = theta + deg2rad(90);
            obj.LEFT_LINE    = line([x - obj.cartWidth/2, x - obj.cartWidth/2], [0, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.RIGHT_LINE   = line([x + obj.cartWidth/2, x + obj.cartWidth/2], [0, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.TOP_LINE     = line([x - obj.cartWidth/2, x + obj.cartWidth/2], [obj.cartHeight, obj.cartHeight], 'Parent', obj.guiAxes);
            obj.BOT_LINE     = line([x - obj.cartWidth/2, x + obj.cartWidth/2], [0, 0], 'Parent', obj.guiAxes);
            obj.POLE_LINE    = line([x, x + obj.poleLength*cos(thetaUp)], [obj.cartHeight/2, (obj.cartHeight/2 + obj.poleLength)*sin(thetaUp)], 'Parent', obj.guiAxes);
            
            
            obj.ANGLE_DISPLAY = uicontrol('Parent', obj.fig, 'Style', 'edit', 'String', -1*theta*180/pi);
%             uicontrol('Parent', obj.fig, 'Style', 'edit', 'String', x, 'Position', [0.9 0.2 0.1 0.1]);
            
            pause(0.0001)
        end
        
    end
end
        
