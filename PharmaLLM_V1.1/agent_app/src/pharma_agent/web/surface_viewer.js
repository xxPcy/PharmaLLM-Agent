"use strict";
// Software 3D renderer: projects the user's physical-coordinate mesh, no decorative anatomy or WebGL dependency.
window.SurfaceViewer = class SurfaceViewer {
  constructor(host) {
    this.host = host; this.faces = []; this.enabled = null; this.yaw = -.4; this.pitch = -.3; this.zoom = 1; this.frame = null;
    this.canvas = document.createElement("canvas"); this.canvas.setAttribute("aria-label", "分割结果三维表面，可拖动旋转和滚轮缩放"); this.canvas.tabIndex = 0;
    host.append(this.canvas); this.observer = new ResizeObserver(() => this.schedule()); this.observer.observe(host);
    let drag = null;
    this.canvas.onpointerdown = e => { if (e.button !== 0) return; drag = [e.clientX,e.clientY,this.yaw,this.pitch]; this.canvas.setPointerCapture(e.pointerId); };
    this.canvas.onpointermove = e => { if (!drag) return; this.yaw = drag[2] + (e.clientX-drag[0])*.012; this.pitch = Math.max(-1.5,Math.min(1.5,drag[3]+(e.clientY-drag[1])*.012)); this.schedule(); };
    this.canvas.onpointerup = this.canvas.onpointercancel = () => { drag = null; };
    this.canvas.addEventListener("wheel", e => { e.preventDefault(); this.zoom = Math.max(.3,Math.min(6,this.zoom*(e.deltaY < 0 ? 1.1 : .9))); this.schedule(); },{passive:false});
    this.canvas.onkeydown = e => { const changes={ArrowLeft:[-.1,0],ArrowRight:[.1,0],ArrowUp:[0,-.1],ArrowDown:[0,.1]};if(changes[e.key]) {e.preventDefault();this.yaw+=changes[e.key][0];this.pitch+=changes[e.key][1];this.schedule();}if(e.key.toLowerCase()==="r")this.reset(); };
  }
  setMesh(mesh) {
    this.faces = []; let min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for (const segment of mesh.segments) {
      for(let i=0;i<segment.quads.length;i+=12){
        const points=[];for(let j=0;j<12;j+=3){const p=segment.quads.slice(i+j,i+j+3);points.push(p);for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}}
        this.faces.push({points,color:segment.color,label:segment.id});
      }
    }
    this.center=min.map((v,k)=>(v+max[k])/2);this.radius=Math.max(...max.map((v,k)=>v-min[k]),1);this.schedule();
  }
  setLabels(labels){this.enabled=new Set(labels);this.schedule();}
  reset(){this.yaw=-.4;this.pitch=-.3;this.zoom=1;this.schedule();}
  schedule(){if(this.frame!==null)return;this.frame=requestAnimationFrame(()=>{this.frame=null;this.draw();});}
  draw(){
    const w=this.host.clientWidth,h=this.host.clientHeight;if(!w||!h)return;
    const ratio=devicePixelRatio||1;this.canvas.width=w*ratio;this.canvas.height=h*ratio;const ctx=this.canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.fillStyle="#0c0f12";ctx.fillRect(0,0,w,h);if(!this.faces.length)return;
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch),scale=Math.min(w,h)*.73*this.zoom/this.radius;
    const project=p=>{const x=p[0]-this.center[0],y=p[1]-this.center[1],z=p[2]-this.center[2];const a=cy*x+sy*y,b=-sy*x+cy*y;return[a,cp*z-sp*b,sp*z+cp*b];};
    const faces=[];
    for(const face of this.faces){if(this.enabled&&!this.enabled.has(face.label))continue;const p=face.points.map(project),a=p[1].map((v,k)=>v-p[0][k]),b=p[2].map((v,k)=>v-p[0][k]);const n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];const length=Math.hypot(...n)||1;
      if(n[2]<=0)continue;
      const light=.35+.65*Math.max(0,(n[0]*-.3+n[1]*.5+n[2]*.8)/length);
      faces.push({p,depth:p.reduce((v,q)=>v+q[2],0)/4,color:face.color.map(v=>Math.round(v*light))});}
    faces.sort((a,b)=>b.depth-a.depth);
    for(const face of faces){ctx.fillStyle=`rgb(${face.color.join(",")})`;ctx.beginPath();face.p.forEach((p,i)=>{const x=w/2+p[0]*scale,y=h/2-p[1]*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.closePath();ctx.fill();}
    this.canvas.dataset.renderedFaces=String(faces.length);
  }
  destroy(){this.observer.disconnect();if(this.frame!==null)cancelAnimationFrame(this.frame);this.canvas.remove();this.faces=[];}
};
