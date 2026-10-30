'use strict';
let laneLearnTimer=null,laneLearning=false;
function laneStatus(text){$('lane-setup-status').textContent=text;}
window.refreshLaneSetup=()=>{
  const lanes=editor.geometry.items.filter(i=>i.kind==='lane');
  setHTML('lane-distances',lanes.map(lane=>`<div class="lane-distance-card"><div><strong style="color:${/^#[\da-f]{6}$/i.test(lane.color)?lane.color:'#7bf4df'}">${esc(lane.name)}</strong><small>${lane.suggestion?esc(lane.suggestion.supporting_tracks)+' observed tracks · review boundaries':'Manual lane'} · ${lane.points.length===4?'START → END':'Needs four corners for speed'}</small></div><label>Start-to-end road distance<input data-lane-distance="${esc(lane.id)}" aria-label="${esc(lane.name)} distance in metres" type="number" min="0.5" max="5000" step="0.1" placeholder="Metres" value="${lane.distance_metres??''}" ${lane.points.length!==4?'disabled':''}></label><button type="button" data-select-lane="${esc(lane.id)}">Adjust lane ↗</button></div>`).join('')||'<p class="muted">Learn lanes from moving traffic, or draw a four-corner lane. No road distance is guessed.</p>');
};
$('lane-distances').onchange=e=>{const id=e.target.dataset.laneDistance;if(!id)return;const lane=editor.geometry.items.find(i=>i.id===id);if(!lane)return;const value=e.target.value.trim();lane.distance_metres=value===''?null:Number(value);updateEditor();};
$('lane-distances').onclick=e=>{const button=e.target.closest('[data-select-lane]');if(button){editor.selected=button.dataset.selectLane;updateEditor();$('stage').scrollIntoView({behavior:'smooth',block:'center'});}};
function stopLaneLearning(){laneLearning=false;clearTimeout(laneLearnTimer);$('learn-lanes').textContent='Learn lanes from traffic';}
async function learnLanes(){
  if(!laneLearning||!editor.active){stopLaneLearning();return;}
  try{
    const result=await api('/api/geometry/lanes/suggest',{});
    if(!laneLearning||!editor.active)return;
    if(result.source_id&&result.source_id!==state?.stream?.stream_id){stopLaneLearning();laneStatus('Camera changed. Learn lanes again for the new source.');return;}
    if(!result.items.length){laneStatus(`Learning traffic paths… ${result.observed_tracks} tracks observed, ${result.usable_paths} straight paths. At least three matching tracks are needed per lane. Empty or turning lanes may need manual drawing.`);laneLearnTimer=setTimeout(learnLanes,8000);return;}
    editor.geometry.items=[...editor.geometry.items.filter(i=>i.kind!=='lane'),...result.items];
    editor.selected=result.items[0].id;stopLaneLearning();updateEditor();
    laneStatus(`${result.items.length} observed lane corridors found. Review START/END positions and enter each measured distance. Adjust the corners if needed. Empty or turning lanes may be missing.`);
  }catch(error){stopLaneLearning();laneStatus(error.message);}
}
$('learn-lanes').onclick=()=>{if(laneLearning){stopLaneLearning();laneStatus('Learning paused. Incoming video and tracking continue.');return;}laneLearning=true;$('learn-lanes').textContent='Cancel learning';laneStatus('Analyzing observed vehicle paths…');learnLanes();};
$('open-lane-speed').onclick=()=>{switchTab('geometry');window.refreshLaneSetup();$('lane-speed-setup').scrollIntoView({behavior:'smooth',block:'center'});if(!editor.geometry.items.some(i=>i.kind==='lane'))$('learn-lanes').click();};
$('apply-lane-speeds').onclick=async()=>{
  const lanes=editor.geometry.items.filter(i=>i.kind==='lane'&&i.distance_metres!=null);
  if(!lanes.length){laneStatus('Enter a measured start-to-end distance for at least one four-corner lane.');return;}
  try{
    editor.geometry=await api('/api/geometry',{...editor.geometry,calibration:null});
    $('calibration').value='';stopLaneLearning();updateEditor();switchTab('monitor');
    toast(`Lane speed timing active for ${lanes.length} lanes. Live km/h estimates appear during travel; completed gate crossings give an average.`);
  }catch(error){laneStatus(error.message);}
};
window.refreshLaneSetup();
