# Container

## 목적 및 명세
소프트웨어 구성 요소와 모든 종속성을 캡슐화하여 모든 호환 런타임이 실행 환경과 컨테이너의 내용에 관계없이 다른 추가 종속성 없이 실행할 수 있도록 하는 것
표준 컨테이너의 명세는 아래를 정의해야 한다.
1. 파일 포맷.
2. 표준 작업의 집합.
3. 실행 환경.

## 5가지 원칙
1. 표준 작업 : 표준 컨테이너 도구를 사용하여 생성, 시작 및 중지할 수 있으며, 표준 파일 시스템 도구를 사용하여 복사 및 스냅샷을 생성할 수 있으며, 표준 네트워크 도구를 사용하여 다운로드 및 업로드할 수 있습니다
2. 콘텐츠 구애받지 않음 : 모든 표준 작업은 콘텐츠에 관계없이 동일한 효과를 갖습니다.
3. 인프라 구애받지 않음 : OCI가 지원하는 모든 인프라에서 실행할 수 있습니다.
4. 자동화를 위한 설계 : 콘텐츠와 인프라에 관계없이 동일한 표준 작업을 제공하기 때문에 표준 컨테이너는 자동화에 매우 적합. 
5. 산업 레벨 딜리버리 : 위에 나열된 모든 속성을 활용하여 Standard Containers는 대기업과 중소기업이 소프트웨어 제공 파이프라인을 간소화하고 자동화할 수 있도록 지원합니다. 

# OCI (Open Container Initiative)
## 배경
2015년 6월 도커, 코어OS, AWS, 구글, 마이크로소프트, IBM 등 주요 플랫폼 벤더들은 애플리케이션의 이식성(Portability) 관점에 컨테이너 형식과 런타임을 중심으로 개방형 산업 표준을 만드는 명확한 목적을 가지고 시작된 프로젝트이다.

### 컨테이너 런타임
“컨테이너”란 개념을 호스트에서 동작 시킬수 있도록 하는 소프트웨어. 컨테이너의 라이프 사이클, 이미지 레지스트리, Pull/Push 등의 기능을 관장한다.
e.g) Docker, Containerd, CRI-O

## 3가지 핵심(3 Specifications)
### 런타임 사양(runtime-spec)
컨테이너의 구성, 실행 환경 및 라이프 사이클을 지정하는 것을 목표로 한다.

**파일 시스템 번들**
특정 방식으로 구성된 파일 집합이며, 모든 표준 작업을 수행하는 모든 호환 런타임에 필요한 모든 데이터와 메타데이터를 포함합니다. 
컨테이너와 해당 구성 데이터가 로컬 파일 시스템에 저장되어 규정을 준수하는 런타임에서 사용될 수 있는 방법들이 정의되어 있어서 컨테이너를 로드하고 실행하는 데 필요한 모든 정보가 들어 있습니다. 
- config.json : 구성 정보가 저장된 json
- 컨테이너의 root 파일 시스템 : 루트 디렉토리가 config.json에 명시되어 있어야 한다.

**런타임 라이프 사이클**
*상태*
```python
{
    "ociVersion": "0.2.0",
    "id": "oci-container1",
    "status": "running",
    "pid": 4422,
    "bundle": "/containers/redis",
    "annotations": {
        "myKey": "myValue"
    }
}
```

*라이프 사이클*
컨테이너가 생성되는 순간부터 더 이상 존재하지 않는 순간까지 발생하는 이벤트의 타임라인을 의미한다.
create→prestart→createRuntime→createContainer→start→startContainer→postStart→postStop
[보다 자세한 사항](https://github.com/opencontainers/runtime-spec/blob/main/config.md#posix-platform-hooks)

*명령*
create : `create <container-id> <path-to-bundle>` 새 컨테이너를 만든다.
start : `start <container-id>` process 에 명시된 대로 프로그램을 실행한다.
kill : `kill <container-id> <signal>` 컨테이너를 종료한다.
delete : `delete <container-id>` 컨테이너를 삭제한다.

**Config.json**
컨테이너에 대한 표준 작업을 구현하는 데 필요한 메타데이터가 들어 있습니다 . 여기에는 실행할 프로세스, 주입할 환경 변수, 사용할 샌드박싱 기능 등이 포함됩니다.
버전 : Open Container Initiative Runtime Specification의 버전
```python
"ociVersion": "0.1.0”
```

루트 :  컨테이너의 루트 파일 시스템
```python
"root": {
    "path": "rootfs",
    "readonly": true
}
```

마운트 : 루트 디렉토리에 마운트할 path 지정. destination, source 등으로 이루어짐.
```python
"mounts": [
    {
        "destination": "/tmp",
        "type": "tmpfs",
        "source": "tmpfs",
        "options": ["nosuid","strictatime","mode=755","size=65536k"]
    },
    {
        "destination": "/data",
        "type": "none",
        "source": "/volumes/testing",
        "options": ["rbind","rw"]
    }
]
```

프로세스 : 유저 정보 및 컨테이너에서 동작할 작업 관련 정의
```python
"process": {
    "terminal": true,
    "consoleSize": {
        "height": 25,
        "width": 80
    },
    "user": {
        "uid": 1,
        "gid": 1,
        "umask": 63,
        "additionalGids": [5, 6]
    },
    "env": [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "TERM=xterm"
    ],
    "cwd": "/root",
    "args": [
        "sh"
    ],
    ...
  }
```

훅 : POSIX 플랫폼의 경우 컨테이너 라이프사이클에 맞추어 커스텀한 액션을 수행 할 수 있다.
```python
"hooks": {
    "prestart": [
        {
            "path": "/usr/bin/fix-mounts",
            "args": ["fix-mounts", "arg1", "arg2"],
            "env":  [ "key1=value1"]
        },
        {
            "path": "/usr/bin/setup-network"
        }
    ],
    "createRuntime": [
        {
            "path": "/usr/bin/fix-mounts",
            "args": ["fix-mounts", "arg1", "arg2"],
            "env":  [ "key1=value1"]
        },
        {
            "path": "/usr/bin/setup-network"
        }
    ],
    "createContainer": [
        {
            "path": "/usr/bin/mount-hook",
            "args": ["-mount", "arg1", "arg2"],
            "env":  [ "key1=value1"]
        }
    ],
    "startContainer": [
        {
            "path": "/usr/bin/refresh-ldcache"
        }
    ],
    "poststart": [
        {
            "path": "/usr/bin/notify-start",
            "timeout": 5
        }
    ],
    "poststop": [
        {
            "path": "/usr/sbin/cleanup.sh",
            "args": ["cleanup.sh", "-f"]
        }
    ]
}
```

어노테이션 : 추가적인 메타데이터 정보 기입
```python
"annotations": {
    "com.example.gpu-cores": "2"
}
```

**호스트 네임 : 컨테이너 안에서 동작하는 프로세스가 보는 컨테이너의 호스트이름**

**도메인 이름 : 컨테이너 안에서 동작하는 프로세스가 보는 컨테이너의 도메인 이름**
참고 : https://github.com/opencontainers/runtime-spec/blob/main/config.md

**Feature Structure : Container Runtime Caller(도커)에게 Runtime이 제공하는 기술들을 명세해놓은 것.**
e.g.) 버전 및 사용가능한 훅 명세
```python
{
  "ociVersionMin": "1.0.0",
  "ociVersionMax": "1.1.0"
}

"hooks": [
  "prestart",
  "createRuntime",
  "createContainer",
  "startContainer",
  "poststart",
  "poststop"
]
```

### 이미지 사양(image-spec)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de5aa077-a815-45d6-bfda-fd7eb6822ad4/efcc6a02-ebaf-4b51-99cd-a2bb20c1ee8f/Untitled.png)
OCI 이미지의 목적은 컨테이너 이미지를 빌드, 전송 및 준비하여 실행하기 위한 상호 운용 가능한 도구를 만드는 것입니다.
빌드, 전송 및 실행하기 위한 이미지의 콘텐츠 및 종속성에 대한 메타데이터와 최종 실행 가능한 파일 시스템을 구성하기 위해 압축된 파일 시스템 레이어의 ID가 포함합니다.

**이미지 사양의 상위 수준 구성 요소**
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de5aa077-a815-45d6-bfda-fd7eb6822ad4/769be279-4a80-4952-ac13-5fd575cfc9fa/Untitled.png)

**이미지 매니페스트 : 컨테이너 이미지를 구성하는 구성 요소를 정의**
특정 아키텍처와 운영 체제에 대한 단일 컨테이너 이미지에 대한 구성과 레이어 세트를 제공
1. 이미지의 구성을 해시하여 이미지와 해당 구성 요소에 대한 고유 ID를 생성할 수 있도록 합니다.
2. 플랫폼별 이미지 버전에 대한 이미지 매니페스트를 참조하는 "fat 매니페스트"를 통해 다중 아키텍처 이미지를 허용하는 것입니다. 
3. OCI 런타임 사양 으로 변환할 수 있는 것입니다 .
```python
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:b5b2b2c507a0944348e0303114d8d93aaaa081732b86451d9bce1f432a537bc7",
    "size": 7023
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:9834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af107ee8f0",
      "size": 32654
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:3c3a4604a545cdc127456d94e421cd355bca5b528f4a9c1905b15da2eb4a4c6b",
      "size": 16724
    },
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:ec4b8955958665577945c89419d1af06b5f7636b4ac3da7f12184802ad867736",
      "size": 73109
    }
  ],
  "subject": {
    "mediaType": "application/vnd.oci.image.manifest.v1+json",
    "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
    "size": 7682
  },
  "annotations": {
    "com.example.key1": "value1",
    "com.example.key2": "value2"
  }
}
```

**이미지 인덱스 :  특정 이미지 매니페스트를 가리키는 상위 레벨 매니페스트로, 하나 이상의 플랫폼에 이상적**
두 플랫폼의 이미지 매니페스트를 가리키는 간단한 이미지 인덱스를 보여주는 예
```python
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.index.v1+json",
  "manifests": [ # image manifest spec!
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7143,
      "digest": "sha256:e692418e4cbaf90ca69d05a66403747baa33ee08806650b51fab815ad7fc331f",
      "platform": {
        "architecture": "ppc64le",
        "os": "linux"
      }
    },
    {
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "size": 7682,
      "digest": "sha256:5b0bcabd1ed22e9fb1310cf6c2dec7cdef19f0ad69efa1f392e94a4333501270",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    }
  ],
  "annotations": {
    "com.example.key1": "value1",
    "com.example.key2": "value2"
  }
}
```

**이미지 레이아웃 : 이미지의 내용을 나타내는 파일 시스템 레이아웃**
CAS Blobs과 위치 주소 지정 참조(이미지 인덱스)에 대한 디렉토리 구조
blob :  해시 알고리즘에 대한 디렉토리로 구성되며 하위에는 실제 콘텐츠(파일)이 저장되어 있음.
oci-layout : 이미지 레이아웃의 버전을 제공
index.json : 이미지 인덱스
```bash
$ cd example.com/app/
$ find . -type f
./index.json
./oci-layout
./blobs/sha256/3588d02542238316759cbf24502f4344ffcc8a60c803870022f335d1390c13b4
./blobs/sha256/4b0bc1c4050b03c95ef2a8e36e25feac42fd31283e8c30b3ee5df6b043155d3c
./blobs/sha256/7968321274dc6b6171697c33df7815310468e694ac5be0ec03ff053bb135e768
```

**파일 시스템 계층 : 하나 이상의 레이어를 서로 위에 적용하여 완전한 파일 시스템을 만듭니다.** 
파일의 변경이나 삭제도 블롭화 하여 레이어를 만들고 이런 레이어들을 서로 위에 적용하여 완전한 파일 시스템을 만들도록 합니다. 

**변환**
1. 파일 시스템 계층 디렉토리에서 루트 파일 시스템을 추출합니다 .
2. 이미지 구성 blob 을 OCI 런타임 구성 blob으로 변환합니다 .

### 배포 사양(distribution-spec)
콘텐츠 배포를 용이하게 하고 표준화하기 위한 API 프로토콜을 정의한다.

**Pull**
매니페스트와 하나 이상의 블롭이라는 두 가지 구성 요소를 검색하는 것을 중심으로 진행됩니다.
일반적으로 객체를 끌어오는 첫 번째 단계는 매니페스트를 검색하는 것입니다. 그러나 레지스트리에서 어떤 순서로든 콘텐츠를 검색할 수 있습니다.
존재하는 blob 또는 매니페스트 URL에 대한 HEAD 요청은 반드시  `200 OK` 를 반환해야 합니다. 성공적인 응답은 헤더에 업로드된 blob의 다이제스트와 바이트 단위 크기를 포함해야 합니다 (`Docker-Content-Digest`, `Content-Length`.
레지스트리에서 Blob 또는 매니페스트를 찾을 수 없는 경우, 응답 코드는 반드시 `404 Not Found`. 이어야 합니다.
1. Pulling Manifest : `GET`다음 형식으로 URL에 대한 요청을 수행합니다. `/v2/<name>/manifests/<reference>`
2. Pulling Blobs : `GET`다음 형식으로 URL에 대한 요청을 수행합니다. `/v2/<name>/blobs/<digest>`

**Push**
이미지를 구성하는 Blobs이 먼저 업로드되고 Manifest가 이후에 업로드됩니다.

**Pushing blobs**
Chunk로 Push 하거나 Monolithic 하게 Push하는 두 가지 방법이 있습니다.

**Pushing a blob monolithically**
Monolithic한 Push에는 두 가지 방법이 있습니다.:
1. Post 후 Put 요청 하기
    1. 세션 ID(업로드 URL = location) 얻기 (Post : /v2/<name>/blobs/uploads/)
    2. 해당 URL에 블롭을 업로드 (Put : <location>?digest=<digest>), 헤더 설정 :  `Content-Length`, `Content-Type: application/octet-stream`
2. Post 단일 요청
    1. Post :  /v2/<name>/blobs/uploads/?digest=<digest>, 헤더 설정 :  `Content-Length`, `Content-Type: application/octet-stream`
    
**Pushing a blob in chunks**
1. 세션 ID(업로드 URL) 얻기 ( `POST`)
2. 청크 업로드 ( `PATCH`)
3. 세션을 닫습니다 ( `PUT`)

**Pushing Manifests**
매니페스트를 푸시하려면 `PUT` 형식으로  `/v2/<name>/manifests/<reference>`경로에 대한 요청을 수행하고 `Content-Type: application/vnd.oci.image.manifest.v1+json` 헤더를 설정해야 합니다.

**Listing Tags**
태그 목록을 가져오려면 `GET`형식으로 `/v2/<name>/tags/list`경로에 대한 요청을 수행합니다. 
`<name>`는 저장소의 네임스페이스이고, `<tag1>`, `<tag2>`, 는 `<tag3>`각각 저장소의 태그입니다.
```python
{
  "name": "<name>",
  "tags": [
    "<tag1>",
    "<tag2>",
    "<tag3>"
  ]
}
```
[더 자세한 사항](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)

## Container Runtime Interface (CRI)
CRI(Container Runtime Interface)는 클러스터 컴포넌트를 재컴파일할 필요 없이 kubelet이 다양한 컨테이너 런타임을 사용할 수 있도록 하는 플러그인 인터페이스입니다.
클러스터의 각 노드에 작업 중인 컨테이너 런타임이 있어야 kubelet이 파드(Pods)와 해당 컨테이너를 실행할 수 있습니다.
Container Runtime Interface(CRI)는 kubelet과 컨테이너 런타임 간의 통신을 위한 주요 프로토콜입니다.

### OCI vs CRI
- OCI의 목표는 **컨테이너 포맷과 런타임의 표준화**하여, 다양한 컨테이너 기술들이 일관된 방식으로 동작하도록 하는 것이다.
- CRI는 쿠버네티스와 연계된 개념으로, **컨테이너 런타임이 쿠버네티스의 kubelet과 어떻게 통신해야 하는지 정의한 인터페이스**이다.

# Build Tool 비교
BuildKit ( = Docker), Buildah, Kaniko

## Common 
OCI 규격을 따른다. 
Multi-Stage Build가 가능ㅇ하다.

## Multi-Stage Parallel Build ( = Performance 향상)
BuildKit : O
Kaniko : X
Buildah : X 

## Rootless ( = Security)
BuildKit : O / seccomp와 AppArmor를 Disable해야 하기 때문에 Linux Kernal 단에서 보안 취약점을 노출하게 된다는 단점이 존재
Kaniko : X but privileged X 
Buildah : O
> `Buildah` specializes in building OCI images. Buildah's commands replicate all of the commands that are found in a Dockerfile. This allows building images with and without Dockerfiles while not requiring any root privileges.
> 

## Multi-Arch
BuildKit : O
Kaniko : X 
> *본 이미지의 파일 시스템을 추출합니다(Dockerfile의 FROM 이미지). 그런 다음 Dockerfile의 명령을 실행하여 각 명령 후에 사용자 공간에서 파일 시스템을 스냅샷으로 만듭니다. 각 명령 후에 변경된 파일 계층을 기본 이미지에 추가하고(있는 경우) 이미지 메타데이터를 업데이트합니다.*
> 
Kaniko는 모든 것을 자체 컨테이너에 빌드합니다. 즉, Kaniko는 본질적으로 가상화할 수 없고 항상 실행 중인 아키텍처에 대한 컨테이너 이미지를 빌드합니다. 굳이 구현을 원한다면 서로 다른 arch 에서 만든 이미지를 arch tag를 이용해서 푸쉬해야 한다.
Buildah : O