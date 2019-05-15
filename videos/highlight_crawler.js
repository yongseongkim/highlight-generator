// [LCK youtube page](https://www.youtube.com/channel/UCw1DsweY9b2AKGjV4kGJP1A/videos)에서 하이라이트를 크롤링
var teamNames = {
	'샌드박스': 'sb',
	'SB': 'sb',
	'KT': 'kt',
	'아프리카': 'af',
	'AF': 'af',
	'그리핀': 'grf',
	'GRF': 'grf',
	'담원': 'dwg',
	'DWG': 'dwg',
	'한화생명': 'hle',
	'HLE': 'hle',
	'SKT': 'skt',
	'젠지': 'gen',
	'GEN': 'gen',
	'킹존': 'kz',
	'KZ': 'kz',
	'진에어': 'jag',
	'JAG': 'jag'
}
var videos = {}
var items = $('#items')

function scrollToBottom() {
    window.scrollTo(0, items.scrollHeight);
}

function findTeam(teamNames, title) {
    var matched = title.match(/[가-힣a-zA-Z]* vs [가-힣a-zA-Z]*/g);
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
        title = title.replace(/\.\.\./g, '');
        var dateRegexResult = /[0-9]*\.[0-9]*\.[0-9]*/.exec(title);
        var team = findTeam(teamNames, title);
        if (!dateRegexResult || !team || team.length < 2 || title.includes('인터뷰')) {
            continue;
        }
        var date = dateRegexResult[0].replace(/\./g, '');
        var matchName = date + '_' + team[0] + team[1];
        if (!nov[matchName]) {
            nov[matchName] = 0;
        }
        nov[matchName] = nov[matchName] + 1;
        matchName = matchName + '_' + nov[matchName];
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
