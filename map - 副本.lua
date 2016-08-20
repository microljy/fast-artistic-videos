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
	
	-----dot test-----
	--[[
	
	local x=1
	local y=360
	local dot=dotcopy(img,x,y)
	local last_dot=dotbuild(last,x,y)
	local last_styledot=dotbuild(laststyle,x,y)
	
	print(dot)
	print(last_dot)
	print(last_styledot)
	dot_style=dotmap(dot,last_dot,last_styledot)
	print(dot_style)
	]]
	local size=8
	local sample_ratio=1/16
	--[[
	local height=img:size(2)
	local width=img:size(3)
	local minsize=math.max(height,width)/4
	
	last= image.scale(last,minsize, 'bilinear')
	laststyle= image.scale(laststyle,minsize, 'bilinear')
	]]
	start_time = os.time()
	local map=mapbuild(last,laststyle,size,sample_ratio)
	end_time = os.time()
	elapsed_time = os.difftime(end_time-start_time)
	print("Build Running time: " .. elapsed_time .. "s")
	
	
	
	-----compute-----
	start_time = os.time()
	out=map2(img,map,size)
	end_time = os.time()
	elapsed_time = os.difftime(end_time-start_time)
	print("Map Running time: " .. elapsed_time .. "s")
	--out=map(last,laststyle,img)
	

	-----save-----
	out = preprocess(out):float()
	local massfileName =string.format('%sout_%04d.png',save_folder,frameIdx)
	save_image(out,massfileName)
	
	
	

	
end



function mapbuild(last,last_style,size,sample_ratio)
	---init---
	local map=torch.Tensor(3,size,size,size):zero()
	local mapnum=torch.Tensor(size,size,size):zero()
	local height=last:size(2)
	local width=last:size(3)
	local sample=torch.Tensor(height,width):zero()
	----然后不断产生随机数  
	local sample_num=0
	local num=height*width*sample_ratio
	
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
			----easy---
			--[[
				r=math.ceil(last[1][y][x]*size)
				g=math.ceil(last[2][y][x]*size)
				b=math.ceil(last[3][y][x]*size)
				if r==0 then
					r=1
				end
				if g==0 then
					g=1
				end
				if b==0 then
					b=1 
				end
				map[1][r][g][b]=map[1][r][g][b]+last_style[1][y][x]
				map[2][r][g][b]=map[2][r][g][b]+last_style[2][y][x]
				map[3][r][g][b]=map[3][r][g][b]+last_style[3][y][x]
				mapnum[r][g][b]=mapnum[r][g][b]+1
			]]
			
			----complex----
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

function map2(img,map,size)
	local height=img:size(2)
	local width=img:size(3)
	for x=1,width do
		for y=1,height do
			---easy---
			--[[
			r=math.ceil(img[1][y][x]*size)
			g=math.ceil(img[2][y][x]*size)
			b=math.ceil(img[3][y][x]*size)
			if r==0 then
				r=1
			end
			if g==0 then
				g=1
			end
			if b==0 then
				b=1 
			end
			img[1][y][x]=map[1][r][g][b]
			img[2][y][x]=map[2][r][g][b]
			img[3][y][x]=map[3][r][g][b]
			]]
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

function map1(last,last_style,img)
	local sizescale=4
	local channel=img:size(1)
	local height=img:size(2)
	local width=img:size(3)
	local out=torch.Tensor(3,height,width):zero()
	local minsize=11
	local ratio=math.max(height,width)/minsize
	
	minlast= image.scale(last,minsize, 'bilinear')
	minlast_style= image.scale(last_style,minsize, 'bilinear')
	local minheight=minlast:size(2)
	local minwidth=minlast:size(3)
	for i=1,minwidth do
		for j=1,minheight do
			local last_dot=dotbuild(minlast,i,j)
			local last_styledot=dotbuild(minlast_style,i,j)
			for x=math.ceil((i-1)*ratio)+1,math.ceil(i*ratio) do
				for y=math.ceil((j-1)*ratio)+1,math.ceil(j*ratio) do
					local dot=dotcopy(img,x,y)
					dot_style=dotmap(dot,last_dot,last_styledot)
					out[1][y][x]=dot_style[1]
					out[2][y][x]=dot_style[2]
					out[3][y][x]=dot_style[3]
				end
			end
		end
	end
	
	
	return out
	--ize=math.max(height,width)	
	--img = image.scale(img,size, 'bilinear')
	--src = image.scale(src,size, 'bilinear')

	--out = image.scale(out,size*sizescale, 'bilinear')

end


function dotmap(dot,last_dot,last_styledot)
	local function diff(dot1,dot2)
		local sum=0
			for i=1,3 do
			sum=sum+math.abs(dot1[i]-dot2[i])
		end
		return sum
	end
	
	local num=last_dot:size(1)
	local w_sum=0
	local result=torch.Tensor(3):zero()
	for i=1,num do
		temp=diff(dot,last_dot[i])
		if temp==0 then
			return last_styledot[i]
		end
		w_sum=w_sum+1/temp
		
		for j=1,3 do
			result[j]=result[j]+last_styledot[i][j]/temp
		end
	end
	
	for j=1,3 do
		result[j]=result[j]/w_sum
	end
	
	return result
end

function dotcopy(img,x,y)
	local dot=torch.Tensor(3):zero()
	for i=1,3 do
		dot[i]=img[i][y][x]
	end
	return dot
end

function dotbuild(img,x,y)
	local height=img:size(2)
	local width=img:size(3)
	local boder=0
	
	if x==1 or x==width then
		boder=boder+1
	end
	if y==1 or y==height then
		boder=boder+1
	end
	
	local num=9
	if boder==1 then
		num=6
	elseif boder==2 then
		num=4
	end
	
	local dot=torch.Tensor(num,3):zero()
	num=1
	
	if 1 then
		for xx=x-1,x+1 do
			for yy=y-1,y+1 do
				if xx>0 and xx<=width and yy>0 and yy<=height then
					for i=1,3 do
						dot[num][i]=img[i][yy][xx]
					end
					num=num+1
				end
			end
		end
	end
	
	return dot
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


main(params)