// Capture timestamps, startup latency, jitter and unit handling for live labels.
const assert=require('node:assert/strict');
global.window={};
require('../static/js/overlay.js');
const speed=window.VehicleOverlay.speedReading;
const history=[0,.08,.19,.31,.45,.58,.70].map(t=>[1700000000+t,.2+.1*t,.5]);
const reading=speed({history,speed_kmh:null},1000,500);
assert.equal(reading.ready,true);
assert.ok(Math.abs(reading.value-100)<.001);
assert.equal(reading.unit,'px/s');
assert.equal(speed({history:history.slice(0,3)},1000,500).ready,false);
assert.equal(speed({history:[]},1000,500).ready,false);
const noisy=history.map((p,i)=>[p[0],p[1]+(i%2?.001:-.001),p[2]]);
assert.ok(Math.abs(speed({history:noisy},1000,500).value-100)<3);
assert.equal(speed({history,speed_kmh:43.2},1000,500).unit,'km/h*');
// Previous checkpoint averages must not freeze the approaching car's label.
assert.ok(Math.abs(speed({history,speed_px_s:900},1000,500).value-100)<.001);
console.log('Live speed label checks passed');
const calibrated={uid:'network-gap-fixture',speed_calibrated:true,speed_kmh:36,history:[[10,.2,.5]]};
assert.equal(window.VehicleOverlay.speedReading(calibrated,1000,500).value,36);
assert.equal(window.VehicleOverlay.speedReading({...calibrated,speed_kmh:null,history:[[10.4,.3,.5]]},1000,500).value,36);
assert.equal(window.VehicleOverlay.speedReading({...calibrated,speed_kmh:null,history:[[11,.4,.5]]},1000,500).ready,false);
assert.equal(window.VehicleOverlay.speedReading({speed_calibrated:true,speed_kmh:null,history:[[1,.4,.5]]},1000,500).ready,false);
// A decoded frame reuses the most recent observation; speed regression must not
// repeat at display refresh rate. A new observation or frame size invalidates it.
const measure=window.VehicleOverlay.measureSpeed;
let fits=0;
window.VehicleOverlay.measureSpeed=(...args)=>{fits++;return measure(...args);};
const stable={history,speed_kmh:null};
for(let i=0;i<144;i++)speed(stable,1000,500);
assert.equal(fits,1);
speed({...stable},1000,500);
assert.equal(fits,2);
speed(stable,2000,1000);
assert.equal(fits,3);
window.VehicleOverlay.measureSpeed=measure;
console.log('Speed observation caching and unavailable-calibration checks passed');
