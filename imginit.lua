require 'torch'
require 'nn'
require 'image'
require 'map'
require 'loadcaffe'
require 'artistic_video_core'

local flowFile = require 'flowFileLoader'

local call_num=0
map0={}

local function main(params)
	local flow_pattern='run/flow_default/backward_[%d]_{%d}.flo'
	local folder='run/'
	local save_folder='test/'
	--local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
	local frameIdx=4
	
	for frameIdx=4,4 do
		local flowFileName = getFormatedFlowFileName(flow_pattern, math.abs(frameIdx - 1), math.abs(frameIdx))
		print(string.format('Reading flow file "%s".', flowFileName))
		local flow = flowFile.load(flowFileName)
		
		local fileName =string.format('%sframe_%04d.ppm',folder,frameIdx-1)
		print(string.format('Reading image file "%s".', fileName))
		last=image.load(fileName, 3)
		
		local fileName =string.format('%sframe_%04d.ppm',folder,frameIdx)
		print(string.format('Reading image file "%s".', fileName))
		now=image.load(fileName, 3)
		
		
		last_warped = warpImageBlank(last, flow)
		

		local sizescale=2
		local channel=last_warped:size(1)
		local height=last_warped:size(2)/sizescale
		local width=last_warped:size(3)/sizescale
		
		size=math.max(height,width)
		last_warped_s = image.scale(last_warped,size, 'bilinear')
		now_s = image.scale(now,size, 'bilinear')
		
		--compute
		local part=torch.Tensor(3,height,width):zero()
		local sim_thresh=0.2
		for x=1,width do
			for y=1,height do
				if similarjudge(last_warped_s,now_s,x,y,sim_thresh)==0 then
					part[1][y][x]=1
					part[2][y][x]=1
					part[3][y][x]=1
				end
			end
		end
		part = image.scale(part,size*sizescale, 'bilinear')
		

		
		--MAP
		local fileName =string.format('%sout-%04d.png',folder,frameIdx-1)
		print(string.format('Reading image file "%s".', fileName))
		last_style=image.load(fileName, 3)
		
		local fileName =string.format('%sinit-%04d.png',folder,frameIdx)
		print(string.format('Reading image file "%s".', fileName))
		now_init=image.load(fileName, 3)
		
		local map_size=16
		local sample_ratio=1/16
		
		map0=map_build(last,last_style,map_size,sample_ratio)
		
		start_time = os.time()
		
		map_out=color_map_partly(now,now_init,part[1],map0,map_size)
		
		end_time = os.time()
		elapsed_time = os.difftime(end_time-start_time)
		print("Map Running time: " .. elapsed_time .. "s")
		
		
		--SAVE
		part = preprocess(part):float()
		local fileName =string.format('%smass_%04d_%04d.png',save_folder,frameIdx-1,frameIdx)
		save_image(part,fileName)
		
		last_warped = preprocess(last_warped):float()
		local fileName =string.format('%swarped_%04d.png',save_folder,frameIdx)
		save_image(last_warped,fileName)
		
		last = preprocess(last):float()
		local fileName =string.format('%sframe_%04d.png',save_folder,frameIdx-1)
		save_image(last,fileName)
		map_out = preprocess(map_out):float()
		local fileName =string.format('%smyinit_%04d.png',save_folder,frameIdx)
		save_image(map_out,fileName)
		
	end
	
end

function myinit(last,flow,now,last_style,weightsFileName,sizescale,sim_thresh,map_size,sample_ratio,map_cycle)
	local last_warped = warpImage(last, flow)
	local now_init = warpImage(last_style, flow)
	local channel=last_warped:size(1)
	local height=last_warped:size(2)/sizescale
	local width=last_warped:size(3)/sizescale
	
	weight=image.load(weightsFileName,3)
	local saveFileName = string.gsub(weightsFileName,'.pgm','.png')
	image.save(saveFileName,weight)
	
	size=math.max(height,width)
	last_warped_s = image.scale(last_warped,size, 'bilinear')
	
	now_s = image.scale(now,size, 'bilinear')
	
	--compute
	local part=torch.Tensor(3,height,width):zero()
	local cnt=0
	for x=1,width do
		for y=1,height do
			if similarjudge(last_warped_s,now_s,x,y,sim_thresh)==0 then
				part[1][y][x]=1
				part[2][y][x]=1
				part[3][y][x]=1
				cnt=cnt+1
			end
		end
	end
	part = image.scale(part,size*sizescale, 'bilinear')
	

	
	--MAP
	dots=now_init:size(2)*now_init:size(3)
	print(cnt/dots)
	if cnt>dots/1000 and cnt<dots/5 then
		if call_num==0 or map0==nil then
			start_time = os.time()
			map0=map_build(last,last_style,map_size,sample_ratio)
			end_time = os.time()
			elapsed_time = os.difftime(end_time-start_time)
			print("Map Building Running time: " .. elapsed_time .. "s")
		end
		
		call_num=call_num+1
		if call_num>=map_cycle then
			call_num=0
		end
		
		start_time = os.time()
		
		now_init=color_map_partly(now,now_init,part[1],weight[1],map0,map_size)
		
		end_time = os.time()
		elapsed_time = os.difftime(end_time-start_time)
		print("Map Running time: " .. elapsed_time .. "s")
	end
	return now_init
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
  img = img:index(1, perm):div(256.0)
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
--main(params)

function warpImage(img, flow)
  local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
  result = image.warp(img, flow, 'bilinear', true, 'pad', -1)
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = mean_pixel[1]
        result[2][x][y] = mean_pixel[2]
        result[3][x][y] = mean_pixel[3]
      end
    end
  end
  return result
end