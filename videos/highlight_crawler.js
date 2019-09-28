// [Onivia League of Legends Highlights (LCS, LCK, LPL, LMS)](https://www.youtube.com/channel/UCPhab209KEicqPJFAk9IZEA)
// 2019 lck spring highlights https://www.youtube.com/watch?v=AQcz1wk9Kd0&list=PLx1tUfSuJjy3Xf_F-xRbO8zRSww7pmM0Z
// 2019 lck summer highlights https://www.youtube.com/watch?v=1xcWcjQ5EhQ&list=PLx1tUfSuJjy0ZiFSheym97uktIsFUFr2b

var teamNames = {
	'SB': 'sb',
	'KT': 'kt',
    'AF': 'af',
    'AFS': 'af',
	'GRF': 'grf',
    'DWG': 'dwg',
    'DMW': 'dwg',
	'HLE': 'hle',
	'SKT': 'skt',
	'GEN': 'gen',
	'KZ': 'kz',
	'JAG': 'jag'
}
var videos = {}

function getTeam(teamNames, title) {
    var regexResults = /([a-zA-Z]*) vs ([a-zA-Z]*)/.exec(title)
    try {
        var team1 = teamNames[regexResults[1]]
        var team2 = teamNames[regexResults[2]]
        return [team1, team2]
    } catch (e) {
        return null
    }
}

function videoURLs() {
    var nov = {}    // 매치를 구분하기 위해 number of video 를 저장한다.
    var items = document.getElementsByTagName('ytd-playlist-panel-video-renderer')
    var urls = {};
    for(element of items) {
        for (span of element.getElementsByTagName("span")) {
            if (span.id == "video-title") {
                var title = span.title
                var team = getTeam(teamNames, title)
                var game = /Game ([0-9])/.exec(title)
                var week = /W([0-9]*)D[0-9]*/.exec(title)
                if (week != null && week.length > 1)  {
                    week = week[1]
                } else if (title.includes("Finals")) {
                    week = "finals"
                } else if (title.includes("Playoffs")) {
                    week = "playoffs"
                }
                if (game == null || game.length < 2 || week == null) {
                    continue
                }
                var matchName = "lckspringsplit_" + team[0] + "_" + team[1] + "_week_" + week + "_game_" + game[1]
                // var matchName = "lcksummersplit_" + team[0] + "_" + team[1] + "_week_" + week + "_game_" + game[1]
                for (atag of element.getElementsByTagName("a")) {
                    if (atag.id == "wc-endpoint") {
                        url = atag.href
                        break
                    }
                }
                urls[matchName] = url;
                break
            }
        }
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

videos = videoURLs()