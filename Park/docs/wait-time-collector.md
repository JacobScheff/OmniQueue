# Disneyland wait-time collector

The Vercel function `POST /api/collect` stores a snapshot of Disneyland Park attractions from [ThemeParks.wiki](https://www.themeparks.wiki/api) in Cloud Firestore. It checks the park's published schedule before fetching rides. Outside published operating, extra, or event hours, the route returns `park_closed` and writes zero documents. Published schedules determine park closure, so an unannounced park-wide closure during scheduled hours can still produce a snapshot.

## Storage format

`wait_time_snapshots` holds one document per park per five-minute UTC bucket. The document ID contains the park ID and the bucket start. The fields are `sampled_at` (Firestore timestamp), `park_date` (`YYYY-MM-DD` in Disneyland's timezone), `catalog_version`, and `rides` (Firestore bytes). The first successful write for a bucket wins, including when calls overlap.

`wait_time_catalogs/{catalog_version}` contains the corresponding ordered list of attraction IDs and names. The collector creates a new catalog version when the attraction list or names change. Each snapshot retains its reference to the catalog used to write it. Only attractions present in the feed are recorded.

For catalog `encoding_version` 1, each ride takes three bytes in catalog order. Decode the two-byte little-endian unsigned standby wait and the one-byte status with `struct.iter_unpack("<HB", snapshot["rides"])` in Python. Wait `65535` means unavailable and `0` means a reported zero-minute wait. The status values are `0` for `UNKNOWN`, `1` for `OPERATING`, `2` for `DOWN` (temporarily unavailable), `3` for `CLOSED`, and `4` for `REFURBISHMENT`. Non-operating attractions always have an unavailable wait. The snapshot timestamp applies to every ride in the document.

## Setup

1. In the Firebase Console, create or choose a project and open **Firestore Database → Create database**. Create the `(default)` database in **Standard edition** and **Production mode**. The collector creates its collections on the first successful write. [Firestore setup](https://firebase.google.com/docs/firestore/quickstart)
2. In the same project's Google Cloud Console, go to **IAM & Admin → Service Accounts**. Create a dedicated collector account with the **Cloud Datastore User** role. Under **Keys**, select **Add key → Create new key → JSON**. Keep the downloaded key out of the repository. [Firestore IAM](https://firebase.google.com/docs/firestore/security/iam), [service account keys](https://docs.cloud.google.com/iam/docs/keys-create-delete)
3. In Firestore **Indexes → Single field**, add index exemptions for `rides`, `catalog_version`, and `park_date` in `wait_time_snapshots`, and for `attractions` in `wait_time_catalogs`. Keep `sampled_at` indexed for time range queries. The exemptions keep each 50-ride snapshot below the 1 KiB storage budget. [Index exemptions](https://firebase.google.com/docs/firestore/query-data/indexing)
4. Import OmniQueue into Vercel as a separate project with root directory `Park/wait_time_collector` and the **Other** framework preset. Add `FIRESTORE_SERVICE_ACCOUNT_JSON` with the full contents of the downloaded JSON key and `CRON_SECRET` with a long random token. Set both for **Production** and deploy. [Vercel project settings](https://vercel.com/docs/project-configuration/project-settings)
5. In the cron-job.org console, create a job for `https://<production-domain>/api/collect`. Set it to **Every 5 minutes** (`*/5 * * * *`), select **POST** with an empty body, and add the `Authorization` header with value `Bearer <CRON_SECRET>`. [cron-job.org console](https://console.cron-job.org/jobs)
6. Run a test request during published park hours. HTTP 200 with `status` of `collected` and `inserted` of `1` indicates a new snapshot. A repeat call in the same bucket reports `0`. `park_closed` always reports `0`. Check `wait_time_snapshots` and the matching catalog in the Firebase Console. Upstream and database failures return non-2xx responses.
