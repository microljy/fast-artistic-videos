require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'artistic_video_core'

local flowFile = require 'flowFileLoader'

local cmd = torch.CmdLine()
cmd:option('-folder', 'run/')
cmd:option('-frameIdx', 53)
cmd:option('-flow_pattern', 'run/flow_default/backward_[%d]_{%d}.flo',
           'Optical flow files pattern')
		   
local function main(params)
	local flow_pattern='run/flow_default/backward_[%d]_{%d}.flo'
	local flow_pattern2='run/flow_default/forward_[%d]_{%d}.flo'
	local folder='run/'
	local save_folder='test/'
	--local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
	local frameIdx=4
	
	for frameIdx=4,4 do
		local flowFileName = getFormatedFlowFileName(flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
		print(string.format('Reading flow file "%s".', flowFileName))
		local flow = flowFile.load(flowFileName)
		print(flow:size())
		b=flow:size(2)
		c=flow:size(3)
		s=math.max(b,c)*2
		flow[1]=image.scale(flow[1],s, 'bilinear')
		
		img=torch.Tensor(3,b,c):zero()
		img[1]=flow[1]
		img[2]=flow[2]
		img=deprocess(img)-0.5
		print(img:max())
		print(img:min())
		local outfileName =string.format('%sbackward_%04d_%04d.png',save_folder,frameIdx,frameIdx-1)
		image.save(outfileName,img)
		
		
		
				local flowFileName = getFormatedFlowFileName(flow_pattern2, math.abs(frameIdx), math.abs(frameIdx-1))
		print(string.format('Reading flow file "%s".', flowFileName))
		local flow = flowFile.load(flowFileName)
		print(flow:size())
		b=flow:size(2)
		c=flow:size(3)
		
		img=torch.Tensor(3,b,c):zero()
		img[1]=flow[1]
		img[2]=flow[2]
		img=deprocess(img)-0.5
		print(img:max())
		print(img:min())
		local outfileName =string.format('%sforward_%04d_%04d.png',save_folder,frameIdx,frameIdx-1)
		image.save(outfileName,img)
		
	end
	
end

function similarjudge(img,src,x,y,thresh)
	if distincion(img,src,x,y,x,y)<thresh then
		return 1
	end
	if 0 then
		for xx=x-1,x+1 do
			for yy=y-1,y+1 do
				if xx>0 and xx<=src:size(3) and yy>0 and yy<=src:size(2) then
					if distincion(img,src,x,y,xx,yy)<thresh then
						return 1
					end
				end
			end
		end
	end
	return 0
end

function distincion(img,src,x,y,x_src,y_src)
	local sum=0
	for i=1,3 do
		sum=sum+math.abs(img[i][y][x]-src[i][y_src][x_src])
	end
	return sum
end


function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(128.0)
  return img
end

function save_image(img, fileName)
  local disp = deprocess(img:double())
  disp = image.minmax{tensor=disp, min=0, max=1}
  image.save(fileName, disp)
end

function warpImageBlank(img, flow)
  result = image.warp(img, flow, 'bilinear', true, 'pad', -1)
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = 1
        result[2][x][y] = 1
        result[3][x][y] = 1
      end
    end
  end
  return result
end
main(params)