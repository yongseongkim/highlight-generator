// [Lol Esports VOD youtube page](https://www.youtube.com/channel/UCzAypSoOFKCZUts3ULtVT_g/videos)에서 풀영상을 크롤링
var teamNames = {
    'SB': 'sb',
    'sb': 'sb',
    'KT': 'kt',
    'kt': 'kt',
    'AF': 'af',
    'af': 'af',
    'GRF': 'grf',
    'grf': 'grf',
    'DWG': 'dwg',
    'dwg': 'dwg',
    'HLE': 'hle',
    'hle': 'hle',
    'SKT': 'skt',
    'skt': 'skt',
    'GEN': 'gen',
    'gen': 'gen',
    'KZ': 'kz',
    'kz': 'kz',
    'JAG': 'jag',
    'jag': 'jag'
}
var videos = {}
var items = $('#items')

function scrollToBottom() {
    window.scrollTo(0, items.scrollHeight);
}

function findTeam(teamNames, title) {
    var matched = title.toLowerCase().replace(/\./g, '').match(/[a-zA-Z]* vs [a-zA-Z]*/g);
    if (matched) {
        for (let word of matched) {
            var vs = word.split('vs');
            var t1 = teamNames[vs[0].replace(/\s/g, '')];
            var t2 = teamNames[vs[1].replace(/\s/g, '')];
            if (t1 && t2) {
                return [t1, t2]
            }
        }
    }
    return null
}

function videoURLs() {
    var nov = {}    // 매치를 구분하기 위해 number of video 를 저장한다.
    var grids = items.getElementsByTagName('ytd-grid-video-renderer');
    var urls = {};
    for (let item of grids) {
        url = item.getElementsByTagName('a')[0].href;
        title = item.getElementsByTagName('a')[1].title;
        var team = findTeam(teamNames, title);
        var matchRegex = /Week [0-9]* Game [0-9]*/.exec(title);
        if (!matchRegex) {
            matchRegex = /WILDCARD Game [0-9]*/.exec(title);
        }
        if (!matchRegex) {
            matchRegex = /PLAYOFF [a-zA-Z0-9\s]* Game [0-9]*/.exec(title);
        }
        if (!matchRegex) {
            matchRegex = /FINALS Game [0-9]*/.exec(title);
        }
        if (!matchRegex) {
            matchRegex = / Game [0-9]*/.exec(title);
        }
        if (!team || team.length < 2 || !matchRegex || !title.includes('LCK')) {
            continue;
        }
        var matchName = team[0] + '_' + team[1] + '_' + matchRegex[0].toLowerCase().replace(/\s/g, "_")
        matchName = matchName.replace(/(\_)\1/g, "_")
        urls[matchName] = url;
    }
    return urls;
}

function downloadObjectAsJson(exportObj, exportName){
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

scrollToBottom();
// https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
new ResizeObserver(function () {
    videos = videoURLs();
    scrollToBottom();
    }).observe(items)
