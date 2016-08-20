require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'artistic_video_core'


local cmd = torch.CmdLine()
cmd:option('-folder', 'run/')
cmd:option('-frameIdx', 53)
cmd:option('-flow_pattern', 'run/flow_default/backward_[%d]_{%d}.flo',
           'Optical flow files pattern')
		   
local function main(params)
	local folder='run/'
	local save_folder='map/'
	local frameIdx=4
	
	-----read-----
	
	local lastfileName =string.format('%sframe_%04d.ppm',folder,frameIdx-1)
	print(string.format('Reading image file "%s".', lastfileName))
	last=image.load(lastfileName, 3)
	
	local stylefileName =string.format('%sout-%04d.png',folder,frameIdx-1)
	print(string.format('Reading image file "%s".', stylefileName))
	laststyle=image.load(stylefileName, 3)
	
	local fileName =string.format('%sframe_%04d.ppm',folder,frameIdx)
	print(string.format('Reading image file "%s".', fileName))
	img=image.load(fileName, 3)
	
	local map_size=8
	local sample_ratio=1/16

	start_time = os.time()
	local map=map_build(last,laststyle,map_size,sample_ratio)
	end_time = os.time()
	elapsed_time = os.difftime(end_time-start_time)
	print("Build Running time: " .. elapsed_time .. "s")
	
	-----compute-----
	start_time = os.time()
	out=color_map(img,map,map_size)
	end_time = os.time()
	elapsed_time = os.difftime(end_time-start_time)
	print("Map Running time: " .. elapsed_time .. "s")
	--out=map(last,laststyle,img)
	
	-----save-----
	out = preprocess(out):float()
	local massfileName =string.format('%sout_%04d.png',save_folder,frameIdx)
	save_image(out,massfileName)
	
end


function map_build(last,last_style,size,sample_ratio)
	---init---
	local map=torch.Tensor(3,size,size,size):zero()
	local mapnum=torch.Tensor(size,size,size):zero()
	local height=last:size(2)
	local width=last:size(3)
	local sample=torch.Tensor(height,width):zero()
	local sample_num=0
	local num=height*width*sample_ratio
	if num>5000 then
		num=5000
	end
	if sample_ratio>=1 then
		return nil
	end
	
	--get random sample--
	math.randomseed(os.time()) 
	while (sample_num<num)
	do
		x=math.random(width)
		y=math.random(height)
		if sample[y][x]==0 then
			sample[y][x]=1
			sample_num=sample_num+1
		end
	end
	
	----get map----
	for x=1,last:size(3) do
		for y=1,last:size(2) do
		if sample[y][x]==1 then
			local r=math.ceil(last[1][y][x]*size)
			local g=math.ceil(last[2][y][x]*size)
			local b=math.ceil(last[3][y][x]*size)
			local wr=r-last[1][y][x]*size
			local wg=g-last[2][y][x]*size
			local wb=b-last[3][y][x]*size
			for i=r-1,r do
				for j=g-1,g do
					for k=b-1,b do
						if i>=1 and j>=1 and k>=1 and i<=size and j<=size and k<=size then
							local wi,wj,wk
							if i==r then 
								wi=1-wr
							else 
								wi=wr
							end
							if j==g then 
								wj=1-wg
							else 
								wj=wg
							end
							if k==b then 
								wk=1-wb
							else 
								wk=wb
							end
							local w=wi*wj*wk
							mapnum[i][j][k]=mapnum[i][j][k]+w
							map[1][i][j][k]=map[1][i][j][k]+last_style[1][y][x]*w
							map[2][i][j][k]=map[2][i][j][k]+last_style[2][y][x]*w
							map[3][i][j][k]=map[3][i][j][k]+last_style[3][y][x]*w
						end
					end
				end
			end
		end
		
		end
	end
	
	for i=1,size do
		for j=1,size do
			for k=1,size do
				if mapnum[i][j][k]==0 then
					mapnum[i][j][k]=1
					map[1][i][j][k]=(i-0.5)/size
					map[2][i][j][k]=(j-0.5)/size
					map[3][i][j][k]=(k-0.5)/size
				end
			end
		end
	end
	map[1]=map[1]:cdiv(mapnum)
	map[2]=map[2]:cdiv(mapnum)
	map[3]=map[3]:cdiv(mapnum)
	
	return map

end

function color_map(img,map,size)
	local height=img:size(2)
	local width=img:size(3)
	for x=1,width do
		for y=1,height do
		
			local r=math.ceil(img[1][y][x]*size)
			local g=math.ceil(img[2][y][x]*size)
			local b=math.ceil(img[3][y][x]*size)
			local wr=r-img[1][y][x]*size
			local wg=g-img[2][y][x]*size
			local wb=b-img[3][y][x]*size
			local w_sum=0
			img[1][y][x]=0
			img[2][y][x]=0
			img[3][y][x]=0
			
			for i=r-1,r do
				for j=g-1,g do
					for k=b-1,b do
						if i>=1 and j>=1 and k>=1 and i<=size and j<=size and k<=size then
							local wi,wj,wk
							if i==r then 
								wi=1-wr
							else 
								wi=wr
							end
							if j==g then 
								wj=1-wg
							else 
								wj=wg
							end
							if k==b then 
								wk=1-wb
							else 
								wk=wb
							end
							local w=wi*wj*wk
							w_sum=w_sum+w
							img[1][y][x]=img[1][y][x]+map[1][i][j][k]*w
							img[2][y][x]=img[2][y][x]+map[2][i][j][k]*w
							img[3][y][x]=img[3][y][x]+map[3][i][j][k]*w
						end
					end
				end
			end
			img[1][y][x]=img[1][y][x]/w_sum
			img[2][y][x]=img[2][y][x]/w_sum
			img[3][y][x]=img[3][y][x]/w_sum
		end
	end
	return img
end

function color_map_partly(src,dst,part,weight,map,size)
	local height=src:size(2)
	local width=src:size(3)
	for x=1,width do
		for y=1,height do
		
			--if part[y][x]==1 and weight[y][x]==0 then
			if part[y][x]==1 then
				local r=math.ceil(src[1][y][x]*size)
				local g=math.ceil(src[2][y][x]*size)
				local b=math.ceil(src[3][y][x]*size)
				local wr=r-src[1][y][x]*size
				local wg=g-src[2][y][x]*size
				local wb=b-src[3][y][x]*size
				local w_sum=0
				dst[1][y][x]=0
				dst[2][y][x]=0
				dst[3][y][x]=0
				
				for i=r-1,r do
					for j=g-1,g do
						for k=b-1,b do
							if i>=1 and j>=1 and k>=1 and i<=size and j<=size and k<=size then
								local wi,wj,wk
								if i==r then 
									wi=1-wr
								else 
									wi=wr
								end
								if j==g then 
									wj=1-wg
								else 
									wj=wg
								end
								if k==b then 
									wk=1-wb
								else 
									wk=wb
								end
								local w=wi*wj*wk
								w_sum=w_sum+w
								dst[1][y][x]=dst[1][y][x]+map[1][i][j][k]*w
								dst[2][y][x]=dst[2][y][x]+map[2][i][j][k]*w
								dst[3][y][x]=dst[3][y][x]+map[3][i][j][k]*w
							end
						end
					end
				end
			if w_sum>0 then
				dst[1][y][x]=dst[1][y][x]/w_sum
				dst[2][y][x]=dst[2][y][x]/w_sum
				dst[3][y][x]=dst[3][y][x]/w_sum
			end
			
			end
			
		end
	end
	return dst
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


--main(params)