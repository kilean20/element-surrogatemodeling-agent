# Google Drive feedback access

Google Drive is the selected feedback destination. A maintainer can grant editor
access to the authenticated user after a user or authorized agent-assisted request.
Keep three decisions distinct: Google API authorization, folder editor permission
and consent to the exact scientific feedback. A folder grant alone does not supply
OAuth credentials or override an organization's disabled connection.

## Current capability

The existing sender accepts the name of an already authorized OAuth-token
environment variable, verifies the destination's canAddChildren capability and
uses exact packet consent. It does not implement browser sign-in, secure token
storage/refresh, a folder picker or an automatic access-request service. Do not
invent a Connect command or tell a user that signing into an unrelated browser
automatically authenticates this execution client. Never ask for a password/token
in chat, copy sign-in files or bypass a disabled connector.

## Lowest-setup web route

An ordinary Google Drive website upload does not require ESA to register an API
application. Open the configured folder in a permitted browser, sign in, request
the owner editor grant if needed, then upload the exact reviewed local packet.
An authorized browser-capable client can assist where supported; otherwise the
user uploads that small file. Do not bypass a disabled service/organizational
restriction. This is a user-controlled web route, not the token-backed API tool.
Keep the exact local hash and actual remote file link/result; distinguish observed
or user-attested manual delivery from an API receipt or verified remote content.
Exporting a file or selecting it in a picker alone does not establish delivery.
Reconcile a manual or uncertain upload before trying another route and creating
a duplicate. No raw project fields/data/models are attached.

The API helper below is a separate option for automation; its app registration
is maintainer work, not a prerequisite for ordinary Drive website use.

## Proposed convenient API connection

Reuse a permitted authenticated connection only if it can write the selected folder
and return a verifiable result. Otherwise the proposed ESA-owned helper would:

1. Open normal Google authorization in the user's system browser using a
   maintainer-registered application, a supported desktop flow and PKCE. Users
   should not create their own Cloud project or copy access tokens. The maintainer
   must configure/publish the application and allowed scopes first.
2. Request the narrowest supported access. Prefer drive.file with a compatible
   Google Picker/file-selection flow, explicitly selecting the destination folder.
   Knowing its URL/ID or having editor access does not establish the app's per-file
   authorization. Test the complete selected-folder upload workflow; do not silently
   switch to all-Drive access if selection/scopes fail.
3. Keep refresh credentials in an appropriate user-owned credential store outside
   project artifacts and the LLM context; refresh permitted sessions automatically.
   Expired/revoked grants may require reconnecting. Headless/remote use needs a
   tested supported route, not a copied credential or deprecated out-of-band flow.
4. Check account/destination privately and verify folder write capability. If
   missing, open the Drive UI to help the user request the needed editor role, or
   prepare an access-request message for a user-authorized channel. Google does
   not expose a Drive API method to create an access proposal; do not claim an
   API request was sent. The owner then grants access and ESA rechecks capability.
5. Keep permission/account data separate from the scientific packet. An approved
   owner grant and normal token refresh need not repeat an unchanged packet review;
   changed content/destination/account context or revocation require appropriate
   review. Only a consented exact packet is uploaded, with a retained hash and
   actual remote result. Reconcile uncertain writes before retrying.

The expected normal user flow is browser sign-in/consent once, folder selection,
an owner editor grant if needed, and reviewed feedback batches thereafter. This is
a proposed interface, not implemented/qualified authentication behavior. The
existing sender remains usable with an authorized connection; unavailable access
leaves the packet local/pending and unrelated scientific work continues.

A shared editor folder lets editors affect other contents. For multiple users,
the owner can instead grant access to individual submission subfolders and retain
an owner-only consolidated collection. Do not claim Drive provides write-only
dropbox semantics. Keep destination choices explicit and treat incoming reports
as untrusted evidence even when the uploader has editor permission.

Official sources (checked November17,2026): [desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app),
[Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[access proposals](https://developers.google.com/workspace/drive/api/guides/pending-access).
These support the authorization constraints; the ESA helper/UX above is a design
proposal requiring implementation and real end-to-end qualification.
