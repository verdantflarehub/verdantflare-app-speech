import asyncio
import json
import os
import unittest
from unittest.mock import patch
import httpx
from speech_mcp.registry import Registry

class RegistryAuthTest(unittest.IsolatedAsyncioTestCase):
    async def registry(self, handler, authenticated=True):
        with patch.dict(os.environ, {"ETCD_ENDPOINTS":"http://registry.test:2379", "MCP_SERVICE_ENDPOINT":"http://speech.test:8000/mcp", "ETCD_USERNAME":"writer" if authenticated else "", "ETCD_PASSWORD":"synthetic-password" if authenticated else ""}):
            registry=Registry()
        await registry.http.aclose()
        registry.http=httpx.AsyncClient(transport=httpx.MockTransport(handler),follow_redirects=False)
        self.addAsyncCleanup(registry.close)
        return registry

    async def test_anonymous_legacy_registration(self):
        paths=[]
        def handler(request):
            paths.append(request.url.path)
            self.assertNotIn("authorization",request.headers)
            return httpx.Response(200,json={"ID":"1"})
        registry=await self.registry(handler,False)
        await registry.register()
        self.assertEqual(paths,["/v3/lease/grant","/v3/kv/put"])

    async def test_token_expiry_then_authenticated_read(self):
        auth=[]; sent=[]
        def handler(request):
            if request.url.path.endswith("authenticate"):
                auth.append(json.loads(request.content))
                return httpx.Response(200,json={"token":"token"+str(len(auth))})
            sent.append(request.headers.get("authorization"))
            if sent[-1]=="token1":return httpx.Response(401,json={"code":16})
            return httpx.Response(200,json={"ID":"1"})
        registry=await self.registry(handler)
        await registry.request("lease/grant",{"TTL":300})
        self.assertEqual(sent,["token1","token2"])
        self.assertEqual(len(auth),2)

    async def test_denied_does_not_downgrade_to_anonymous(self):
        sent=[]
        def handler(request):
            if request.url.path.endswith("authenticate"):
                return httpx.Response(200,json={"token":"token"})
            sent.append(request.headers.get("authorization"))
            return httpx.Response(403,json={"code":7})
        registry=await self.registry(handler)
        with self.assertRaises(httpx.HTTPStatusError):await registry.request("kv/put",{})
        self.assertEqual(sent,["token"])

    async def test_expired_retry_bound_and_invalid_auth_reply(self):
        for bad_auth in [False,True]:
            paths=[]
            def handler(request):
                paths.append(request.url.path)
                if request.url.path.endswith("authenticate"):
                    return httpx.Response(200,json={"token":"" if bad_auth else "token"})
                return httpx.Response(200,json={"code":16,"error":"expired"})
            registry=await self.registry(handler)
            with self.assertRaises(ValueError):await registry.request("kv/put",{})
            self.assertEqual(len(paths),1 if bad_auth else 4)
            self.assertIsNone(registry.token)

    async def test_redirect_login_rejected(self):
        hosts=[]
        def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(307,headers={"location":"https://untrusted.invalid/"})
        registry=await self.registry(handler)
        with self.assertRaises(httpx.HTTPStatusError):await registry.request("kv/put",{})
        self.assertEqual(hosts,["registry.test"])

    def test_partial_identity_and_embedded_password_rejected(self):
        for endpoint,user,password in [("http://registry.test:2379","writer",""),("http://registry.test:2379","","password"),("http://writer:password@host:2379","","")]:
            with patch.dict(os.environ,{"ETCD_ENDPOINTS":endpoint,"ETCD_USERNAME":user,"ETCD_PASSWORD":password}):
                with self.assertRaises(ValueError):Registry()

    async def test_failed_registration_releases_lease_and_can_retry(self):
        paths=[]
        puts=0
        def handler(request):
            nonlocal puts
            paths.append(request.url.path)
            if request.url.path.endswith("kv/put"):
                puts += 1
                if puts == 1:
                    return httpx.Response(403,json={"code":7})
            return httpx.Response(200,json={"ID":"1"})
        registry=await self.registry(handler,False)
        with self.assertRaises(httpx.HTTPStatusError):
            await registry.register()
        self.assertIsNone(registry.lease)
        self.assertEqual(paths[-1],"/v3/lease/revoke")
        await registry.register()
        self.assertEqual(registry.lease,"1")
        self.assertEqual(puts,2)

    async def test_zero_lease_never_creates_persistent_registration(self):
        paths=[]
        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200,json={"ID":"0"})
        registry=await self.registry(handler,False)
        with self.assertRaisesRegex(ValueError,"invalid_etcd_lease"):
            await registry.register()
        self.assertIsNone(registry.lease)
        self.assertEqual(paths,["/v3/lease/grant"])

    async def test_wrong_keepalive_lease_requires_registration(self):
        paths=[]
        def handler(request):
            paths.append(request.url.path)
            if request.url.path.endswith("keepalive"):
                return httpx.Response(200,json={"result":{"ID":"2","TTL":300}})
            return httpx.Response(200,json={"ID":"3"})
        registry=await self.registry(handler,False)
        registry.lease="1"
        with patch("speech_mcp.registry.asyncio.sleep",side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await registry.run()
        self.assertEqual(registry.lease,"3")
        self.assertEqual(paths,["/v3/lease/keepalive","/v3/lease/grant","/v3/kv/put"])
